from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Iterable,
    Iterator,
    Optional,
    Self,
    Sequence,
    Type,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field

from .messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from .utils.code_agent import CodeExecutionResult, FunctionSignature, get_default_repl_tool

if TYPE_CHECKING:
    from instructor import Partial
    from langchain_experimental.tools.python.tool import PythonAstREPLTool

PydanticModelT = TypeVar("PydanticModelT", bound=BaseModel)
StructuredOutputType: TypeAlias = dict[object, object] | BaseModel

DEFAULT_IMAGE_DESCRIPTION_INSTRUCTION = "Provide a detailed description of all visible elements in the image, summarizing key details in a few clear sentences."
DEFAULT_CODE_GENERATION_PROMPT = (
    "You are utilizing a Python code execution tool now.\n"
    "Your goal is to generate Python code that solves the task efficiently and appends both the code and its output to your context memory.\n"
    "Since your context window is highly limited, type `pass` if no code execution is needed.\n"
    "\n"
    "To optimize tool efficiency, follow these guidelines:\n"
    "- Write concise, efficient code that directly serves the intended purpose.\n"
    "- Avoid unnecessary operations (e.g., excessive loops, recursion, or heavy computations).\n"
    "- Handle potential errors gracefully (e.g., using try-except blocks).\n"
    "- Prevent excessive output by limiting print statements to essential information only (e.g., avoid printing large datasets).\n"
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


class Chatterer(BaseModel):
    """Language model for generating text from a given input."""

    client: BaseChatModel
    structured_output_kwargs: dict[str, Any] = Field(default_factory=dict)

    @overload
    def __call__(
        self,
        messages: LanguageModelInput,
        response_model: Type[PydanticModelT],
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> PydanticModelT: ...

    @overload
    def __call__(
        self,
        messages: LanguageModelInput,
        response_model: None = None,
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str: ...

    def __call__(
        self,
        messages: LanguageModelInput,
        response_model: Optional[Type[PydanticModelT]] = None,
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str | PydanticModelT:
        if response_model:
            return self.generate_pydantic(response_model, messages, config, stop, **kwargs)
        return self.client.invoke(input=messages, config=config, stop=stop, **kwargs).text()

    @classmethod
    def openai(
        cls,
        model: str = "gpt-4o-mini",
        structured_output_kwargs: Optional[dict[str, Any]] = {"strict": True},
    ) -> Self:
        from langchain_openai import ChatOpenAI

        return cls(client=ChatOpenAI(model=model), structured_output_kwargs=structured_output_kwargs or {})

    @classmethod
    def anthropic(
        cls,
        model_name: str = "claude-3-7-sonnet-20250219",
        structured_output_kwargs: Optional[dict[str, Any]] = None,
    ) -> Self:
        from langchain_anthropic import ChatAnthropic

        return cls(
            client=ChatAnthropic(model_name=model_name, timeout=None, stop=None),
            structured_output_kwargs=structured_output_kwargs or {},
        )

    @classmethod
    def google(
        cls,
        model: str = "gemini-2.0-flash",
        structured_output_kwargs: Optional[dict[str, Any]] = None,
    ) -> Self:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return cls(
            client=ChatGoogleGenerativeAI(model=model),
            structured_output_kwargs=structured_output_kwargs or {},
        )

    @classmethod
    def ollama(
        cls,
        model: str = "deepseek-r1:1.5b",
        structured_output_kwargs: Optional[dict[str, Any]] = None,
    ) -> Self:
        from langchain_ollama import ChatOllama

        return cls(
            client=ChatOllama(model=model),
            structured_output_kwargs=structured_output_kwargs or {},
        )

    def generate(
        self,
        messages: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str:
        return self.client.invoke(input=messages, config=config, stop=stop, **kwargs).text()

    async def agenerate(
        self,
        messages: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str:
        return (await self.client.ainvoke(input=messages, config=config, stop=stop, **kwargs)).text()

    def generate_stream(
        self,
        messages: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        for chunk in self.client.stream(input=messages, config=config, stop=stop, **kwargs):
            yield chunk.text()

    async def agenerate_stream(
        self,
        messages: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        async for chunk in self.client.astream(input=messages, config=config, stop=stop, **kwargs):
            yield chunk.text()

    def generate_pydantic(
        self,
        response_model: Type[PydanticModelT],
        messages: LanguageModelInput,
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
        response_model: Type[PydanticModelT],
        messages: LanguageModelInput,
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
        response_model: Type[PydanticModelT],
        messages: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Iterator[PydanticModelT]:
        try:
            import instructor
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
        response_model: Type[PydanticModelT],
        messages: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[PydanticModelT]:
        try:
            import instructor
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
                content=[{"type": "text", "text": instruction}, {"type": "image_url", "image_url": {"url": image_url}}],
            )
        ])

    async def adescribe_image(self, image_url: str, instruction: str = DEFAULT_IMAGE_DESCRIPTION_INSTRUCTION) -> str:
        """
        Create a detailed description of an image using the Vision Language Model asynchronously.
        - image_url: Image URL to describe
        """
        return await self.agenerate([
            HumanMessage(
                content=[{"type": "text", "text": instruction}, {"type": "image_url", "image_url": {"url": image_url}}],
            )
        ])

    @staticmethod
    def get_num_tokens_from_message(message: BaseMessage) -> Optional[tuple[int, int]]:
        try:
            if isinstance(message, AIMessage) and (usage_metadata := message.usage_metadata):
                input_tokens = int(usage_metadata["input_tokens"])
                output_tokens = int(usage_metadata["output_tokens"])
            else:
                # Dynamic extraction for unknown structures
                input_tokens: Optional[int] = None
                output_tokens: Optional[int] = None

                def _find_tokens(obj: object) -> None:
                    nonlocal input_tokens, output_tokens
                    if isinstance(obj, dict):
                        for key, value in cast(dict[object, object], obj).items():
                            if isinstance(value, int):
                                if "input" in str(key) or "prompt" in str(key):
                                    input_tokens = value
                                elif "output" in str(key) or "completion" in str(key):
                                    output_tokens = value
                            else:
                                _find_tokens(value)
                    elif isinstance(obj, list):
                        for item in cast(list[object], obj):
                            _find_tokens(item)

                _find_tokens(message.model_dump())

            if input_tokens is None or output_tokens is None:
                return None
            return input_tokens, output_tokens
        except Exception:
            return None

    def invoke_code_execution(
        self,
        messages: LanguageModelInput,
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

    async def ainvoke_code_execution(
        self,
        messages: LanguageModelInput,
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


class PythonCodeToExecute(BaseModel):
    code: str = Field(description="Python code to execute")


def _with_structured_output(
    client: BaseChatModel,
    response_model: Type["PydanticModelT | Partial[PydanticModelT]"],
    structured_output_kwargs: dict[str, Any],
) -> Runnable[LanguageModelInput, dict[object, object] | BaseModel]:
    return client.with_structured_output(schema=response_model, **structured_output_kwargs)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]


# def _add_message_last(messages: LanguageModelInput, prompt_to_add: str) -> LanguageModelInput:
#     if isinstance(messages, str):
#         messages += f"\n{prompt_to_add}"
#     elif isinstance(messages, Sequence):
#         messages = list(messages)
#         messages.append(SystemMessage(content=prompt_to_add))
#     else:
#         messages = messages.to_messages()
#         messages.append(SystemMessage(content=prompt_to_add))
#     return messages


def _add_message_first(messages: LanguageModelInput, prompt_to_add: str) -> LanguageModelInput:
    if isinstance(messages, str):
        messages = f"{prompt_to_add}\n{messages}"
    elif isinstance(messages, Sequence):
        messages = list(messages)
        messages.insert(0, SystemMessage(content=prompt_to_add))
    else:
        messages = messages.to_messages()
        messages.insert(0, SystemMessage(content=prompt_to_add))
    return messages


def augment_prompt_for_toolcall(
    function_signatures: Iterable[FunctionSignature],
    messages: LanguageModelInput,
    prompt_for_code_invoke: Optional[str] = DEFAULT_CODE_GENERATION_PROMPT,
    function_reference_prefix: Optional[str] = DEFAULT_FUNCTION_REFERENCE_PREFIX_PROMPT,
    function_reference_seperator: str = DEFAULT_FUNCTION_REFERENCE_SEPARATOR,
) -> LanguageModelInput:
    if function_signatures:
        messages = _add_message_first(
            messages=messages,
            prompt_to_add=FunctionSignature.as_prompt(
                function_signatures, function_reference_prefix, function_reference_seperator
            ),
        )
    if prompt_for_code_invoke:
        messages = _add_message_first(messages=messages, prompt_to_add=prompt_for_code_invoke)
    return messages


def interactive_shell(
    chatterer: Chatterer = Chatterer.openai(),
    system_instruction: BaseMessage | Iterable[BaseMessage] = ([
        SystemMessage("You are an AI that can answer questions and execute Python code."),
    ]),
    repl_tool: Optional["PythonAstREPLTool"] = None,
    prompt_for_code_invoke: Optional[str] = DEFAULT_CODE_GENERATION_PROMPT,
    additional_callables: Optional[Callable[..., object] | Sequence[Callable[..., object]]] = None,
    function_reference_prefix: Optional[str] = DEFAULT_FUNCTION_REFERENCE_PREFIX_PROMPT,
    function_reference_seperator: str = DEFAULT_FUNCTION_REFERENCE_SEPARATOR,
    config: Optional[RunnableConfig] = None,
    stop: Optional[list[str]] = None,
    **kwargs: Any,
) -> None:
    from rich.console import Console
    from rich.prompt import Prompt

    # 코드 실행 필요 여부를 판단하는 모델
    class IsCodeExecutionNeeded(BaseModel):
        is_code_execution_needed: bool = Field(
            description="Whether Python tool calling is needed to answer user query."
        )

    # 추가 코드 실행 필요 여부를 판단하는 모델
    class IsFurtherCodeExecutionNeeded(BaseModel):
        review_on_code_execution: str = Field(description="Review on the code execution.")
        next_action: str = Field(description="Next action to take.")
        is_further_code_execution_needed: bool = Field(
            description="Whether further Python tool calling is needed to answer user query."
        )

    # REPL 도구 초기화
    if repl_tool is None:
        repl_tool = get_default_repl_tool()

    function_signatures: list[FunctionSignature] = FunctionSignature.from_callable(additional_callables)
    console = Console()
    context: list[BaseMessage] = []
    if system_instruction:
        if isinstance(system_instruction, BaseMessage):
            context.append(system_instruction)
        else:
            context.extend(system_instruction)

    # 환영 메시지
    console.print("[bold blue]Welcome to the Interactive Chatterer Shell![/bold blue]")
    console.print("Type 'quit' or 'exit' to end the conversation.")

    while True:
        # 사용자 입력 받기
        user_input = Prompt.ask("[bold green]You[/bold green]")
        if user_input.lower() in ["quit", "exit"]:
            console.print("[bold blue]Goodbye![/bold blue]")
            break

        context.append(HumanMessage(content=user_input))

        # 코드 실행 필요 여부 판단
        decision = chatterer.generate_pydantic(
            response_model=IsCodeExecutionNeeded,
            messages=augment_prompt_for_toolcall(
                function_signatures=function_signatures,
                messages=context,
                prompt_for_code_invoke=prompt_for_code_invoke,
                function_reference_prefix=function_reference_prefix,
                function_reference_seperator=function_reference_seperator,
            ),
        )

        # 코드 실행 처리
        if decision.is_code_execution_needed:
            code_result = chatterer.invoke_code_execution(
                messages=context,
                repl_tool=repl_tool,
                prompt_for_code_invoke=prompt_for_code_invoke,
                function_signatures=function_signatures,
                function_reference_prefix=function_reference_prefix,
                function_reference_seperator=function_reference_seperator,
                config=config,
                stop=stop,
                **kwargs,
            )

            if code_result.code.strip() == "pass":
                tool_use_message = None
            else:
                code_session_messages: list[BaseMessage] = []
                while True:
                    code_execution_message = AIMessage(
                        content=f"Executed code:\n```python\n{code_result.code}\n```\nOutput:\n{code_result.output}".strip()
                    )
                    code_session_messages.append(code_execution_message)
                    console.print("[bold yellow]Executed code:[/bold yellow]")
                    console.print(f"[code]{code_result.code}[/code]")
                    console.print("[bold yellow]Output:[/bold yellow]")
                    console.print(code_result.output)

                    decision = chatterer.generate_pydantic(
                        response_model=IsFurtherCodeExecutionNeeded,
                        messages=augment_prompt_for_toolcall(
                            function_signatures=function_signatures,
                            messages=context + code_session_messages,
                            prompt_for_code_invoke=prompt_for_code_invoke,
                            function_reference_prefix=function_reference_prefix,
                            function_reference_seperator=function_reference_seperator,
                        ),
                    )
                    review_on_code_execution = decision.review_on_code_execution.strip()
                    next_action = decision.next_action.strip()
                    console.print("[bold blue]AI:[/bold blue]")
                    console.print(f"-[bold yellow]Review on code execution:[/bold yellow] {review_on_code_execution}")
                    console.print(f"-[bold yellow]Next Action:[/bold yellow] {next_action}")
                    code_session_messages.append(
                        AIMessage(
                            content=f"- Review upon code execution: {review_on_code_execution}\n- Next Action: {next_action}".strip()
                        )
                    )
                    if not decision.is_further_code_execution_needed:
                        tool_use_message = code_execution_message
                        break
        else:
            tool_use_message = None

        # 코드 실행 결과 컨텍스트에 추가
        if tool_use_message:
            context.append(tool_use_message)

        # AI 응답 스트리밍 출력
        console.print("[bold blue]AI:[/bold blue] ", end="")
        response = ""
        for chunk in chatterer.generate_stream(messages=context):
            response += chunk
            console.print(chunk, end="")

        # 전체 응답 처리 후 컨텍스트에 추가
        lines = response.split("\n")
        if lines:
            lines[-1] = lines[-1].rstrip()  # 마지막 줄의 오른쪽 공백 제거
        response = "\n".join(lines).strip()
        context.append(AIMessage(content=response))
        console.print()  # 응답 후 줄바꿈 추가


if __name__ == "__main__":
    interactive_shell()
