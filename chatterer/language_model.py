import os
import re
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Concatenate,
    Iterable,
    Iterator,
    Optional,
    ParamSpec,
    Self,
    Sequence,
    Type,
    TypeAlias,
    TypedDict,
    TypeVar,
    overload,
)

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.utils.utils import secret_from_env
from loguru import logger
from pydantic import BaseModel, Field, SecretStr

from .constants import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_GOOGLE_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_XAI_MODEL,
)
from .messages import AIMessage, BaseMessage, HumanMessage, UsageMetadata
from .utils.code_agent import CodeExecutionResult, FunctionSignature, augment_prompt_for_toolcall

if TYPE_CHECKING:
    from instructor import Partial  # pyright: ignore[reportMissingTypeStubs]

    from .utils.repl_tool import PythonAstREPLTool

P = ParamSpec("P")
PydanticModelT = TypeVar("PydanticModelT", bound=BaseModel)
StructuredOutputType: TypeAlias = dict[object, object] | BaseModel

DEFAULT_IMAGE_DESCRIPTION_INSTRUCTION = "Provide a detailed description of all visible elements in the image, summarizing key details in a few clear sentences."
DEFAULT_CODE_GENERATION_PROMPT = (
    "You are utilizing a Python code execution tool now.\n"
    "Your goal is to generate Python code that solves the task efficiently and appends both the code and its output to your context memory.\n"
    "\n"
    "To optimize tool efficiency, follow these guidelines:\n"
    "- Write concise, efficient code that directly serves the intended purpose.\n"
    "- Avoid unnecessary operations (e.g., excessive loops, recursion, or heavy computations).\n"
    "- Handle potential errors gracefully (e.g., using try-except blocks).\n"
    "\n"
    "Return your response strictly in the following JSON format:\n"
    '{\n  "code": "<your_python_code_here>"\n}\n\n'
)


DEFAULT_FUNCTION_REFERENCE_PREFIX_PROMPT = (
    "Below functions are included in global scope and can be used in your code.\n"
    "Do not try to redefine the function(s).\n"
    "You don't have to force yourself to use these tools - use them only when you need to.\n"
)
DEFAULT_FUNCTION_REFERENCE_SEPARATOR = "\n---\n"  # Separator to distinguish different function references

PYTHON_CODE_PATTERN: re.Pattern[str] = re.compile(r"```(?:python\s*\n)?(.*?)```", re.DOTALL)


class FactoryOption(TypedDict, total=False):
    structured_output_kwargs: dict[str, object]
    api_key: str
    kwargs: dict[str, Any]


ProviderFactory: TypeAlias = Callable[Concatenate[Type[PydanticModelT], P], PydanticModelT]
PROVIDER_FACTORIES: dict[str, ProviderFactory[..., ...]] = {}


def _sanitize_provider(provider: str) -> str:
    return provider.lower().replace(" ", "_").replace("-", "_")


def register_providers(*registry_names: str):
    def inner(impl: ProviderFactory[PydanticModelT, P]):
        def wrapper(cls: Type[PydanticModelT], *args: P.args, **kwargs: P.kwargs) -> PydanticModelT:
            return impl(cls, *args, **kwargs)

        for registry_name in registry_names:
            PROVIDER_FACTORIES[_sanitize_provider(registry_name)] = wrapper
        return wrapper

    return inner


class Chatterer(BaseModel):
    """Language model for generating text from a given input."""

    client: BaseChatModel
    structured_output_kwargs: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_provider(
        cls, provider_and_model: str, option: Optional[FactoryOption] = {"structured_output_kwargs": {"strict": True}}
    ) -> Self:
        backend, model = provider_and_model.split(":", 1)
        if (sanitized_backend := _sanitize_provider(backend)) != backend:
            logger.warning(f"Sanitized provider: {backend} to {sanitized_backend}")
            backend = sanitized_backend
        if func := PROVIDER_FACTORIES.get(backend):
            return func(cls, model, option)
        else:
            raise ValueError(
                f"Unsupported provider: {backend}. Supported providers are: {', '.join(PROVIDER_FACTORIES.keys())}."
            )

    @classmethod
    @register_providers("openai", "oai", "chatgpt")
    def openai(
        cls, model: str = DEFAULT_OPENAI_MODEL, option: FactoryOption = {"structured_output_kwargs": {"strict": True}}
    ) -> Self:
        from langchain_openai import ChatOpenAI

        if TYPE_CHECKING:
            ChatOpenAI(model=model, api_key=SecretStr(""))

        return cls(
            client=ChatOpenAI(
                model=model,
                **_handle_option(option=option, handle_api_key={"env": "OPENAI_API_KEY", "key": "api_key"}),
            ),
            structured_output_kwargs=option.get("structured_output_kwargs", {}),
        )

    @classmethod
    @register_providers("anthropic", "claude")
    def anthropic(cls, model: str = DEFAULT_ANTHROPIC_MODEL, option: FactoryOption = {}) -> Self:
        from langchain_anthropic import ChatAnthropic

        if TYPE_CHECKING:
            ChatAnthropic(model_name=model, api_key=SecretStr(""), timeout=10, stop=["\n"])

        return cls(
            client=ChatAnthropic(
                model_name=model,
                **_handle_option(option=option, handle_api_key={"env": "ANTHROPIC_API_KEY", "key": "api_key"}),
            ),
            structured_output_kwargs=option.get("structured_output_kwargs", {}),
        )

    @classmethod
    @register_providers("google_genai", "google", "genai", "gemini")
    def google_genai(cls, model: str = DEFAULT_GOOGLE_MODEL, option: FactoryOption = {}) -> Self:
        from langchain_google_genai import ChatGoogleGenerativeAI

        if TYPE_CHECKING:
            ChatGoogleGenerativeAI(model=model, api_key=SecretStr(""))

        return cls(
            client=ChatGoogleGenerativeAI(
                model=model,
                **_handle_option(option=option, handle_api_key={"env": "GOOGLE_API_KEY", "key": "api_key"}),
            ),
            structured_output_kwargs=option.get("structured_output_kwargs", {}),
        )

    @classmethod
    @register_providers("google_vertexai", "vertexai", "vertex", "vertex_ai")
    def google_vertexai(cls, model: str, option: FactoryOption = {}) -> Self:
        from langchain_google_vertexai import ChatVertexAI

        if TYPE_CHECKING:
            ChatVertexAI(model=model)

        return cls(
            client=ChatVertexAI(model=model, **_handle_option(option=option)),
        )

    @classmethod
    @register_providers("ollama")
    def ollama(cls, model: str, option: FactoryOption = {}) -> Self:
        from langchain_ollama import ChatOllama

        if TYPE_CHECKING:
            ChatOllama(model=model)

        return cls(
            client=ChatOllama(model=model, **_handle_option(option=option)),
            structured_output_kwargs=option.get("structured_output_kwargs", {}),
        )

    @classmethod
    @register_providers("openrouter", "open_router")
    def openrouter(cls, model: str = DEFAULT_OPENROUTER_MODEL, option: FactoryOption = {}) -> Self:
        from langchain_openai import ChatOpenAI

        if TYPE_CHECKING:
            ChatOpenAI(model=model, api_key=SecretStr(""), base_url="")

        return cls(
            client=ChatOpenAI(
                model=model,
                **_handle_option(
                    option=option,
                    handle_api_key={"env": "OPENROUTER_API_KEY", "key": "api_key"},
                    handle_envs=[
                        {
                            "env": "OPENROUTER_BASE_URL",
                            "key": "base_url",
                            "default": "https://openrouter.ai/api/v1",
                        },
                    ],
                ),
            ),
            structured_output_kwargs=option.get("structured_output_kwargs", {}),
        )

    @classmethod
    @register_providers("xai", "x_ai", "x", "grok")
    def xai(cls, model: str = DEFAULT_XAI_MODEL, option: FactoryOption = {}) -> Self:
        from langchain_openai import ChatOpenAI

        if TYPE_CHECKING:
            ChatOpenAI(model=model, api_key=SecretStr(""), base_url="")

        return cls(
            client=ChatOpenAI(
                model=model,
                **_handle_option(
                    option=option,
                    handle_api_key={"env": "XAI_API_KEY", "key": "api_key"},
                    handle_envs=[{"env": "XAI_BASE_URL", "key": "base_url", "default": "https://api.x.ai/v1"}],
                ),
            ),
            structured_output_kwargs=option.get("structured_output_kwargs", {}),
        )

    @classmethod
    @register_providers("aws", "aws_bedrock", "bedrock", "amazon")
    def aws(cls, model: str, option: FactoryOption = {}) -> Self:
        from langchain_aws import ChatBedrock

        if TYPE_CHECKING:
            ChatBedrock(model=model)

        return cls(
            client=ChatBedrock(model=model, **_handle_option(option=option)),
            structured_output_kwargs=option.get("structured_output_kwargs", {}),
        )

    @classmethod
    @register_providers("huggingface", "hf")
    def huggingface(cls, model: str, option: FactoryOption = {}) -> Self:
        from langchain_huggingface import ChatHuggingFace

        if TYPE_CHECKING:
            ChatHuggingFace(model=model)

        return cls(
            client=ChatHuggingFace(model=model, **_handle_option(option=option)),
            structured_output_kwargs=option.get("structured_output_kwargs", {}),
        )

    @property
    def invoke(self):
        return self.client.invoke

    @property
    def ainvoke(self):
        return self.client.ainvoke

    @property
    def stream(self):
        return self.client.stream

    @property
    def astream(self):
        return self.client.astream

    @property
    def bind_tools(self):  # pyright: ignore[reportUnknownParameterType]
        return self.client.bind_tools  # pyright: ignore[reportUnknownParameterType, reportUnknownVariableType, reportUnknownMemberType]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    @overload
    def __call__(
        self,
        messages: LanguageModelInput,
        *,
        response_model: Type[PydanticModelT],
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> PydanticModelT: ...
    @overload
    def __call__(
        self,
        messages: LanguageModelInput,
        *,
        response_model: None = None,
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str: ...
    def __call__(
        self,
        messages: LanguageModelInput,
        *,
        response_model: Optional[Type[PydanticModelT]] = None,
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str | PydanticModelT:
        if response_model:
            return self.generate_pydantic(messages, response_model=response_model, config=config, stop=stop, **kwargs)
        return self.client.invoke(input=messages, config=config, stop=stop, **kwargs).text

    def generate(
        self,
        messages: LanguageModelInput,
        *,
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str:
        return self.client.invoke(input=messages, config=config, stop=stop, **kwargs).text

    async def agenerate(
        self,
        messages: LanguageModelInput,
        *,
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str:
        return (await self.client.ainvoke(input=messages, config=config, stop=stop, **kwargs)).text

    def generate_stream(
        self,
        messages: LanguageModelInput,
        *,
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        for chunk in self.client.stream(input=messages, config=config, stop=stop, **kwargs):
            yield chunk.text

    async def agenerate_stream(
        self,
        messages: LanguageModelInput,
        *,
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        async for chunk in self.client.astream(input=messages, config=config, stop=stop, **kwargs):
            yield chunk.text

    def generate_pydantic(
        self,
        messages: LanguageModelInput,
        *,
        response_model: Type[PydanticModelT],
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> PydanticModelT:
        result: StructuredOutputType = _with_structured_output(
            client=self.client,
            response_model=response_model,
            structured_output_kwargs=self.structured_output_kwargs,
        ).invoke(input=messages, config=config, stop=stop, **kwargs)
        if isinstance(result, response_model):
            return result
        else:
            return response_model.model_validate(result)

    async def agenerate_pydantic(
        self,
        messages: LanguageModelInput,
        *,
        response_model: Type[PydanticModelT],
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> PydanticModelT:
        result: StructuredOutputType = await _with_structured_output(
            client=self.client,
            response_model=response_model,
            structured_output_kwargs=self.structured_output_kwargs,
        ).ainvoke(input=messages, config=config, stop=stop, **kwargs)
        if isinstance(result, response_model):
            return result
        else:
            return response_model.model_validate(result)

    def generate_pydantic_stream(
        self,
        messages: LanguageModelInput,
        *,
        response_model: Type[PydanticModelT],
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Iterator[PydanticModelT]:
        try:
            import instructor  # pyright: ignore[reportMissingTypeStubs]
        except ImportError:
            raise ImportError("Please install `instructor` with `pip install instructor` to use this feature.")

        partial_response_model = instructor.Partial[response_model]
        for chunk in _with_structured_output(
            client=self.client,
            response_model=partial_response_model,
            structured_output_kwargs=self.structured_output_kwargs,
        ).stream(input=messages, config=config, stop=stop, **kwargs):
            yield response_model.model_validate(chunk)

    async def agenerate_pydantic_stream(
        self,
        messages: LanguageModelInput,
        *,
        response_model: Type[PydanticModelT],
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[PydanticModelT]:
        try:
            import instructor  # pyright: ignore[reportMissingTypeStubs]
        except ImportError:
            raise ImportError("Please install `instructor` with `pip install instructor` to use this feature.")

        partial_response_model = instructor.Partial[response_model]
        async for chunk in _with_structured_output(
            client=self.client,
            response_model=partial_response_model,
            structured_output_kwargs=self.structured_output_kwargs,
        ).astream(input=messages, config=config, stop=stop, **kwargs):
            yield response_model.model_validate(chunk)

    def describe_image(self, image_url: str, instruction: str = DEFAULT_IMAGE_DESCRIPTION_INSTRUCTION) -> str:
        """
        Create a detailed description of an image using the Vision Language Model.
        - image_url: Image URL to describe
        """
        return self.generate([
            HumanMessage(
                content=[
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            )
        ])

    async def adescribe_image(self, image_url: str, instruction: str = DEFAULT_IMAGE_DESCRIPTION_INSTRUCTION) -> str:
        """
        Create a detailed description of an image using the Vision Language Model asynchronously.
        - image_url: Image URL to describe
        """
        return await self.agenerate([
            HumanMessage(
                content=[
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            )
        ])

    def get_approximate_token_count(self, message: BaseMessage) -> int:
        return self.client.get_num_tokens_from_messages([message])  # pyright: ignore[reportUnknownMemberType]

    def get_usage_metadata(self, message: BaseMessage) -> UsageMetadata:
        if isinstance(message, AIMessage):
            usage_metadata = message.usage_metadata
            if usage_metadata is not None:
                input_tokens = usage_metadata["input_tokens"]
                output_tokens = usage_metadata["output_tokens"]
                return {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
            else:
                approx_tokens = self.get_approximate_token_count(message)
                return {"input_tokens": 0, "output_tokens": approx_tokens, "total_tokens": approx_tokens}
        else:
            approx_tokens = self.get_approximate_token_count(message)
            return {
                "input_tokens": approx_tokens,
                "output_tokens": 0,
                "total_tokens": approx_tokens,
            }

    def exec(
        self,
        messages: LanguageModelInput,
        *,
        repl_tool: Optional["PythonAstREPLTool"] = None,
        prompt_for_code_invoke: Optional[str] = DEFAULT_CODE_GENERATION_PROMPT,
        function_signatures: Optional[FunctionSignature | Iterable[FunctionSignature]] = None,
        function_reference_prefix: Optional[str] = DEFAULT_FUNCTION_REFERENCE_PREFIX_PROMPT,
        function_reference_seperator: str = DEFAULT_FUNCTION_REFERENCE_SEPARATOR,
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> CodeExecutionResult:
        if not function_signatures:
            function_signatures = []
        elif isinstance(function_signatures, FunctionSignature):
            function_signatures = [function_signatures]
        messages = augment_prompt_for_toolcall(
            function_signatures=function_signatures,
            messages=messages,
            prompt_for_code_invoke=prompt_for_code_invoke,
            function_reference_prefix=function_reference_prefix,
            function_reference_seperator=function_reference_seperator,
        )
        code_obj: PythonCodeToExecute = self.generate_pydantic(
            response_model=PythonCodeToExecute, messages=messages, config=config, stop=stop, **kwargs
        )
        return CodeExecutionResult.from_code(
            code=code_obj.code,
            config=config,
            repl_tool=repl_tool,
            function_signatures=function_signatures,
            **kwargs,
        )

    @property
    def invoke_code_execution(self) -> Callable[..., CodeExecutionResult]:
        """Alias for exec method for backward compatibility."""
        return self.exec

    async def aexec(
        self,
        messages: LanguageModelInput,
        *,
        repl_tool: Optional["PythonAstREPLTool"] = None,
        prompt_for_code_invoke: Optional[str] = DEFAULT_CODE_GENERATION_PROMPT,
        additional_callables: Optional[Callable[..., object] | Sequence[Callable[..., object]]] = None,
        function_reference_prefix: Optional[str] = DEFAULT_FUNCTION_REFERENCE_PREFIX_PROMPT,
        function_reference_seperator: str = DEFAULT_FUNCTION_REFERENCE_SEPARATOR,
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> CodeExecutionResult:
        function_signatures: list[FunctionSignature] = FunctionSignature.from_callable(additional_callables)
        messages = augment_prompt_for_toolcall(
            function_signatures=function_signatures,
            messages=messages,
            prompt_for_code_invoke=prompt_for_code_invoke,
            function_reference_prefix=function_reference_prefix,
            function_reference_seperator=function_reference_seperator,
        )
        code_obj: PythonCodeToExecute = await self.agenerate_pydantic(
            response_model=PythonCodeToExecute, messages=messages, config=config, stop=stop, **kwargs
        )
        return await CodeExecutionResult.afrom_code(
            code=code_obj.code,
            config=config,
            repl_tool=repl_tool,
            function_signatures=function_signatures,
            **kwargs,
        )

    @property
    def ainvoke_code_execution(self):
        """Alias for aexec method for backward compatibility."""
        return self.aexec


class PythonCodeToExecute(BaseModel):
    code: str = Field(description="Python code to execute")

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)

        codes: list[str] = []
        for match in PYTHON_CODE_PATTERN.finditer(self.code):
            codes.append(match.group(1))
        if codes:
            self.code = "\n".join(codes)


def _with_structured_output(
    client: BaseChatModel,
    response_model: Type["PydanticModelT | Partial[PydanticModelT]"],
    structured_output_kwargs: dict[str, Any],
) -> Runnable[LanguageModelInput, dict[object, object] | BaseModel]:
    return client.with_structured_output(schema=response_model, **structured_output_kwargs)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]


class HandleApiKey(TypedDict):
    env: str
    key: str


class HandleEnv(TypedDict):
    env: str
    key: str
    default: Optional[str]


def _handle_option(
    option: FactoryOption,
    handle_api_key: Optional[HandleApiKey] = None,
    handle_envs: Optional[Iterable[HandleEnv]] = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = option.get("kwargs", {})
    if handle_api_key is not None:
        if (api_key := option.get(handle_api_key["key"])) is None:
            api_key_found: SecretStr | None = secret_from_env(handle_api_key["env"], default=None)()
            if api_key_found is not None:
                kwargs[handle_api_key["key"]] = api_key_found
        else:
            kwargs[handle_api_key["key"]] = SecretStr(api_key)
    if handle_envs is not None:
        for handle_env in handle_envs:
            if (env_value := os.environ.get(handle_env["env"])) is not None:
                kwargs[handle_env["key"]] = env_value
            elif handle_env["default"] is not None:
                kwargs[handle_env["key"]] = handle_env["default"]
    return kwargs
