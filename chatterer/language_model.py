from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Iterator,
    Optional,
    Self,
    Type,
    TypeAlias,
    TypeVar,
)

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from instructor import Partial

PydanticModelT = TypeVar("PydanticModelT", bound=BaseModel)
StructuredOutputType: TypeAlias = dict[object, object] | BaseModel

DEFAULT_IMAGE_DESCRIPTION_INSTRUCTION = "Just describe all the details you see in the image in few sentences."


class Chatterer(BaseModel):
    """Language model for generating text from a given input."""

    client: BaseChatModel
    structured_output_kwargs: dict[str, Any] = Field(default_factory=dict)

    def __call__(self, messages: LanguageModelInput) -> str:
        """
        Generate text from the given input messages.

        Args:
            messages (LanguageModelInput): Input messages for the language model.
                Can be one of the following types:
                - str: A single string message.
                - list[dict[str, str]]: A list of dictionaries with 'role' and 'content' keys.
                - tuple[str, str]: A tuple of strings representing the role and content of a single message.
                - list[BaseMessage]: A list of BaseMessage objects. (BaseMessage is a Pydantic model; e.g. can import AIMessage, HumanMessage, SystemMessage from langchain_core.messages)
        """
        return self.generate(messages)

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
        result: StructuredOutputType = with_structured_output(
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
        result: StructuredOutputType = await with_structured_output(
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
        for chunk in with_structured_output(
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
        async for chunk in with_structured_output(
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


def with_structured_output(
    client: BaseChatModel,
    response_model: Type["PydanticModelT | Partial[PydanticModelT]"],
    structured_output_kwargs: dict[str, Any],
) -> Runnable[LanguageModelInput, dict[object, object] | BaseModel]:
    return client.with_structured_output(schema=response_model, **structured_output_kwargs)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]


if __name__ == "__main__":
    import asyncio

    # 테스트용 Pydantic 모델 정의
    class Propositions(BaseModel):
        proposition_topic: str
        proposition_content: str

    chatterer = Chatterer.openai()
    prompt = "What is the meaning of life?"

    # === Synchronous Tests ===

    # 1. generate
    print("=== Synchronous generate ===")
    result_sync = chatterer.generate(prompt)
    print("Result (generate):", result_sync)

    # 2. __call__
    print("\n=== Synchronous __call__ ===")
    result_call = chatterer(prompt)
    print("Result (__call__):", result_call)

    # 3. generate_stream
    print("\n=== Synchronous generate_stream ===")
    for i, chunk in enumerate(chatterer.generate_stream(prompt)):
        print(f"Chunk {i}:", chunk)

    # 4. generate_pydantic
    print("\n=== Synchronous generate_pydantic ===")
    try:
        result_pydantic = chatterer.generate_pydantic(Propositions, prompt)
        print("Result (generate_pydantic):", result_pydantic)
    except Exception as e:
        print("Error in generate_pydantic:", e)

    # 5. generate_pydantic_stream
    print("\n=== Synchronous generate_pydantic_stream ===")
    try:
        for i, chunk in enumerate(chatterer.generate_pydantic_stream(Propositions, prompt)):
            print(f"Pydantic Chunk {i}:", chunk)
    except Exception as e:
        print("Error in generate_pydantic_stream:", e)

    # === Asynchronous Tests ===

    # Async helper function to enumerate async iterator
    async def async_enumerate(aiter: AsyncIterator[Any], start: int = 0) -> AsyncIterator[tuple[int, Any]]:
        i = start
        async for item in aiter:
            yield i, item
            i += 1

    async def run_async_tests():
        # 6. agenerate
        print("\n=== Asynchronous agenerate ===")
        result_async = await chatterer.agenerate(prompt)
        print("Result (agenerate):", result_async)

        # 7. agenerate_stream
        print("\n=== Asynchronous agenerate_stream ===")
        async for i, chunk in async_enumerate(chatterer.agenerate_stream(prompt)):
            print(f"Async Chunk {i}:", chunk)

        # 8. agenerate_pydantic
        print("\n=== Asynchronous agenerate_pydantic ===")
        try:
            result_async_pydantic = await chatterer.agenerate_pydantic(Propositions, prompt)
            print("Result (agenerate_pydantic):", result_async_pydantic)
        except Exception as e:
            print("Error in agenerate_pydantic:", e)

        # 9. agenerate_pydantic_stream
        print("\n=== Asynchronous agenerate_pydantic_stream ===")
        try:
            i = 0
            async for chunk in chatterer.agenerate_pydantic_stream(Propositions, prompt):
                print(f"Async Pydantic Chunk {i}:", chunk)
                i += 1
        except Exception as e:
            print("Error in agenerate_pydantic_stream:", e)

    asyncio.run(run_async_tests())
