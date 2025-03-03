from typing import (
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
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field

PydanticModelT = TypeVar("PydanticModelT", bound=BaseModel)
ContentType: TypeAlias = str | list[str | dict[str, Any]]
StructuredOutputType: TypeAlias = dict[str, Any] | BaseModel


class Chatterer(BaseModel):
    """Language model for generating text from a given input."""

    client: BaseChatModel
    structured_output_kwargs: dict[str, Any] = Field(default_factory=dict)

    def __call__(self, messages: LanguageModelInput) -> str:
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
        content: ContentType = self.client.invoke(input=messages, config=config, stop=stop, **kwargs).content
        if isinstance(content, str):
            return content
        else:
            return "".join(part for part in content if isinstance(part, str))

    async def agenerate(
        self,
        messages: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str:
        content: ContentType = (await self.client.ainvoke(input=messages, config=config, stop=stop, **kwargs)).content
        if isinstance(content, str):
            return content
        else:
            return "".join(part for part in content if isinstance(part, str))

    def generate_stream(
        self,
        messages: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        for chunk in self.client.stream(input=messages, config=config, stop=stop, **kwargs):
            content: ContentType = chunk.content
            if isinstance(content, str):
                yield content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, str):
                        yield part
                    else:
                        continue
            else:
                continue

    async def agenerate_stream(
        self,
        messages: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        async for chunk in self.client.astream(input=messages, config=config, stop=stop, **kwargs):
            content: ContentType = chunk.content
            if isinstance(content, str):
                yield content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, str):
                        yield part
                    else:
                        continue
            else:
                continue

    def generate_pydantic(
        self,
        response_model: Type[PydanticModelT],
        messages: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> PydanticModelT:
        result: StructuredOutputType = self.client.with_structured_output(
            response_model, **self.structured_output_kwargs
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
        result: StructuredOutputType = await self.client.with_structured_output(
            response_model, **self.structured_output_kwargs
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
        for chunk in self.client.with_structured_output(partial_response_model, **self.structured_output_kwargs).stream(
            input=messages, config=config, stop=stop, **kwargs
        ):
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
        async for chunk in self.client.with_structured_output(
            partial_response_model, **self.structured_output_kwargs
        ).astream(input=messages, config=config, stop=stop, **kwargs):
            yield response_model.model_validate(chunk)


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

    # === Synchronous generate ===
    # Result (generate): The meaning of life is a deeply philosophical question that has been pondered by humans for centuries. Different cultures, religions, and belief systems have their own interpretations of the meaning of life. Some believe that the meaning of life is to seek happiness and pleasure, others believe it is to serve a higher power or fulfill a spiritual purpose. Ultimately, the meaning of life is a personal and subjective concept that each individual must find and define for themselves.

    # === Synchronous __call__ ===
    # Result (__call__): The meaning of life is a philosophical and existential question that has been debated for centuries. Different individuals, cultures, and philosophies offer varying perspectives on the purpose and significance of life. Some believe that the meaning of life is to seek happiness, fulfillment, and personal growth, while others believe it is to fulfill a higher spiritual or moral purpose. Ultimately, the meaning of life is a deeply personal and subjective concept that each individual must explore and discover for themselves.

    # === Synchronous generate_stream ===
    # Chunk 0:
    # Chunk 1: The
    # Chunk 2:  meaning
    # Chunk 3:  of
    # Chunk 4:  life
    # Chunk 5:  is
    # Chunk 6:  a
    # Chunk 7:  complex
    # Chunk 8:  and
    # Chunk 9:  deeply
    # Chunk 10:  personal
    # Chunk 11:  question
    # Chunk 12:  that
    # Chunk 13:  has
    # Chunk 14:  been
    # Chunk 15:  debated
    # Chunk 16:  by
    # Chunk 17:  philosophers
    # Chunk 18: ,
    # Chunk 19:  theolog
    # Chunk 20: ians
    # Chunk 21: ,
    # Chunk 22:  and
    # Chunk 23:  individuals
    # Chunk 24:  throughout
    # Chunk 25:  history
    # Chunk 26: .
    # Chunk 27:  The
    # Chunk 28:  answer
    # Chunk 29:  to
    # Chunk 30:  this
    # Chunk 31:  question
    # Chunk 32:  can
    # Chunk 33:  vary
    # Chunk 34:  greatly
    # Chunk 35:  depending
    # Chunk 36:  on
    # Chunk 37:  one
    # Chunk 38: 's
    # Chunk 39:  beliefs
    # Chunk 40: ,
    # Chunk 41:  values
    # Chunk 42: ,
    # Chunk 43:  and
    # Chunk 44:  experiences
    # Chunk 45: .
    # Chunk 46:  Some
    # Chunk 47:  may
    # Chunk 48:  find
    # Chunk 49:  meaning
    # Chunk 50:  in
    # Chunk 51:  pursuing
    # Chunk 52:  personal
    # Chunk 53:  happiness
    # Chunk 54:  and
    # Chunk 55:  fulfillment
    # Chunk 56: ,
    # Chunk 57:  others
    # Chunk 58:  may
    # Chunk 59:  find
    # Chunk 60:  meaning
    # Chunk 61:  in
    # Chunk 62:  contributing
    # Chunk 63:  to
    # Chunk 64:  the
    # Chunk 65:  greater
    # Chunk 66:  good
    # Chunk 67:  of
    # Chunk 68:  society
    # Chunk 69: ,
    # Chunk 70:  while
    # Chunk 71:  others
    # Chunk 72:  may
    # Chunk 73:  find
    # Chunk 74:  meaning
    # Chunk 75:  in
    # Chunk 76:  spiritual
    # Chunk 77:  or
    # Chunk 78:  religious
    # Chunk 79:  beliefs
    # Chunk 80: .
    # Chunk 81:  Ultimately
    # Chunk 82: ,
    # Chunk 83:  the
    # Chunk 84:  meaning
    # Chunk 85:  of
    # Chunk 86:  life
    # Chunk 87:  is
    # Chunk 88:  subjective
    # Chunk 89:  and
    # Chunk 90:  can
    # Chunk 91:  differ
    # Chunk 92:  from
    # Chunk 93:  person
    # Chunk 94:  to
    # Chunk 95:  person
    # Chunk 96: .
    # Chunk 97:

    # === Synchronous generate_pydantic ===
    # C:\Users\cosogi\chatterer\.venv\Lib\site-packages\langchain_openai\chat_models\base.py:1390: UserWarning: Cannot use method='json_schema' with model gpt-3.5-turbo since it doesn't support OpenAI's Structured Output API. You can see supported models here: https://platform.openai.com/docs/guides/structured-outputs#supported-models. To fix this warning, set `method='function_calling'. Overriding to method='function_calling'.
    # warnings.warn(
    # Result (generate_pydantic): proposition_topic='meaning of life' proposition_content='The meaning of life is a philosophical question that has been debated by thinkers and scholars for centuries. There are different perspectives on the meaning of life, including religious, existential, and philosophical views. Some argue that the meaning of life is to seek happiness and fulfillment, while others believe it is to fulfill a higher purpose or serve others. Ultimately, the meaning of life is subjective and may vary depending on individual beliefs and values.'

    # === Synchronous generate_pydantic_stream ===
    # C:\Users\cosogi\chatterer\.venv\Lib\site-packages\langchain_openai\chat_models\base.py:1390: UserWarning: Cannot use method='json_schema' with model gpt-3.5-turbo since it doesn't support OpenAI's Structured Output API. You can see supported models here: https://platform.openai.com/docs/guides/structured-outputs#supported-models. To fix this warning, set `method='function_calling'. Overriding to method='function_calling'.
    # warnings.warn(
    # Pydantic Chunk 0: proposition_topic='Meaning of Life' proposition_content=''
    # Pydantic Chunk 1: proposition_topic='Meaning of Life' proposition_content='The'
    # Pydantic Chunk 2: proposition_topic='Meaning of Life' proposition_content='The meaning'
    # Pydantic Chunk 3: proposition_topic='Meaning of Life' proposition_content='The meaning of'
    # Pydantic Chunk 4: proposition_topic='Meaning of Life' proposition_content='The meaning of life'
    # Pydantic Chunk 5: proposition_topic='Meaning of Life' proposition_content='The meaning of life is'
    # Pydantic Chunk 6: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a'
    # Pydantic Chunk 7: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical'
    # Pydantic Chunk 8: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and'
    # Pydantic Chunk 9: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential'
    # Pydantic Chunk 10: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question'
    # Pydantic Chunk 11: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that'
    # Pydantic Chunk 12: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has'
    # Pydantic Chunk 13: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been'
    # Pydantic Chunk 14: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated'
    # Pydantic Chunk 15: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by'
    # Pydantic Chunk 16: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars'
    # Pydantic Chunk 17: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars,'
    # Pydantic Chunk 18: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars, philosophers'
    # Pydantic Chunk 19: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars, philosophers,'
    # Pydantic Chunk 20: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars, philosophers, and'
    # Pydantic Chunk 21: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars, philosophers, and individuals'
    # Pydantic Chunk 22: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars, philosophers, and individuals throughout'
    # Pydantic Chunk 23: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars, philosophers, and individuals throughout history'
    # Pydantic Chunk 24: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars, philosophers, and individuals throughout history.'
    # Pydantic Chunk 25: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars, philosophers, and individuals throughout history. It'
    # Pydantic Chunk 26: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars, philosophers, and individuals throughout history. It refers'
    # Pydantic Chunk 27: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars, philosophers, and individuals throughout history. It refers to'
    # Pydantic Chunk 28: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars, philosophers, and individuals throughout history. It refers to the'
    # Pydantic Chunk 29: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars, philosophers, and individuals throughout history. It refers to the purpose'
    # Pydantic Chunk 30: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars, philosophers, and individuals throughout history. It refers to the purpose,'
    # Pydantic Chunk 31: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars, philosophers, and individuals throughout history. It refers to the purpose, significance'
    # Pydantic Chunk 32: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars, philosophers, and individuals throughout history. It refers to the purpose, significance,'
    # Pydantic Chunk 33: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars, philosophers, and individuals throughout history. It refers to the purpose, significance, or'
    # Pydantic Chunk 34: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars, philosophers, and individuals throughout history. It refers to the purpose, significance, or value'
    # Pydantic Chunk 35: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars, philosophers, and individuals throughout history. It refers to the purpose, significance, or value of'
    # Pydantic Chunk 36: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars, philosophers, and individuals throughout history. It refers to the purpose, significance, or value of human'
    # Pydantic Chunk 37: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars, philosophers, and individuals throughout history. It refers to the purpose, significance, or value of human existence'
    # Pydantic Chunk 38: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars, philosophers, and individuals throughout history. It refers to the purpose, significance, or value of human existence and'
    # Pydantic Chunk 39: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars, philosophers, and individuals throughout history. It refers to the purpose, significance, or value of human existence and consciousness'
    # Pydantic Chunk 40: proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that has been debated by scholars, philosophers, and individuals throughout history. It refers to the purpose, significance, or value of human existence and consciousness.'

    # === Asynchronous agenerate ===
    # Result (agenerate): The meaning of life is a complex and subjective philosophical question that has been debated throughout history. Different individuals and cultures may have different beliefs and perspectives on the meaning of life. Some may believe that the meaning of life is to seek happiness and fulfillment, others may believe it is to serve a higher power or contribute to the greater good of society. Ultimately, the meaning of life is a highly personal and individualistic concept that each person must determine for themselves.

    # === Asynchronous agenerate_stream ===
    # Async Chunk 0:
    # Async Chunk 1: This
    # Async Chunk 2:  is
    # Async Chunk 3:  a
    # Async Chunk 4:  deeply
    # Async Chunk 5:  philosophical
    # Async Chunk 6:  question
    # Async Chunk 7:  that
    # Async Chunk 8:  has
    # Async Chunk 9:  puzzled
    # Async Chunk 10:  humans
    # Async Chunk 11:  for
    # Async Chunk 12:  centuries
    # Async Chunk 13: .
    # Async Chunk 14:  The
    # Async Chunk 15:  meaning
    # Async Chunk 16:  of
    # Async Chunk 17:  life
    # Async Chunk 18:  is
    # Async Chunk 19:  a
    # Async Chunk 20:  highly
    # Async Chunk 21:  individual
    # Async Chunk 22:  and
    # Async Chunk 23:  subjective
    # Async Chunk 24:  concept
    # Async Chunk 25: ,
    # Async Chunk 26:  with
    # Async Chunk 27:  different
    # Async Chunk 28:  people
    # Async Chunk 29:  finding
    # Async Chunk 30:  meaning
    # Async Chunk 31:  in
    # Async Chunk 32:  different
    # Async Chunk 33:  things
    # Async Chunk 34: .
    # Async Chunk 35:  Some
    # Async Chunk 36:  may
    # Async Chunk 37:  find
    # Async Chunk 38:  meaning
    # Async Chunk 39:  in
    # Async Chunk 40:  their
    # Async Chunk 41:  relationships
    # Async Chunk 42: ,
    # Async Chunk 43:  their
    # Async Chunk 44:  work
    # Async Chunk 45: ,
    # Async Chunk 46:  their
    # Async Chunk 47:  beliefs
    # Async Chunk 48: ,
    # Async Chunk 49:  or
    # Async Chunk 50:  their
    # Async Chunk 51:  experiences
    # Async Chunk 52: .
    # Async Chunk 53:  Others
    # Async Chunk 54:  may
    # Async Chunk 55:  find
    # Async Chunk 56:  meaning
    # Async Chunk 57:  in
    # Async Chunk 58:  the
    # Async Chunk 59:  pursuit
    # Async Chunk 60:  of
    # Async Chunk 61:  personal
    # Async Chunk 62:  growth
    # Async Chunk 63: ,
    # Async Chunk 64:  knowledge
    # Async Chunk 65: ,
    # Async Chunk 66:  or
    # Async Chunk 67:  happiness
    # Async Chunk 68: .
    # Async Chunk 69:  Ultimately
    # Async Chunk 70: ,
    # Async Chunk 71:  the
    # Async Chunk 72:  meaning
    # Async Chunk 73:  of
    # Async Chunk 74:  life
    # Async Chunk 75:  is
    # Async Chunk 76:  something
    # Async Chunk 77:  that
    # Async Chunk 78:  each
    # Async Chunk 79:  person
    # Async Chunk 80:  must
    # Async Chunk 81:  discover
    # Async Chunk 82:  for
    # Async Chunk 83:  themselves
    # Async Chunk 84:  through
    # Async Chunk 85:  intros
    # Async Chunk 86: pection
    # Async Chunk 87: ,
    # Async Chunk 88:  reflection
    # Async Chunk 89: ,
    # Async Chunk 90:  and
    # Async Chunk 91:  experience
    # Async Chunk 92: .
    # Async Chunk 93:

    # === Asynchronous agenerate_pydantic ===
    # C:\Users\cosogi\chatterer\.venv\Lib\site-packages\langchain_openai\chat_models\base.py:1390: UserWarning: Cannot use method='json_schema' with model gpt-3.5-turbo since it doesn't support OpenAI's Structured Output API. You can see supported models here: https://platform.openai.com/docs/guides/structured-outputs#supported-models. To fix this warning, set `method='function_calling'. Overriding to method='function_calling'.
    # warnings.warn(
    # Result (agenerate_pydantic): proposition_topic='Meaning of Life' proposition_content='The meaning of life is a philosophical and existential question that explores the purpose and significance of human existence. It is a complex and subjective concept that has been debated by philosophers, theologians, and thinkers throughout history. Some believe that the meaning of life is to seek happiness, fulfillment, or spiritual enlightenment, while others argue that it is to make a positive impact on the world or to find meaning in personal relationships and experiences. Ultimately, the meaning of life is a deeply personal and reflective question that each individual must contemplate and define for themselves.'

    # === Asynchronous agenerate_pydantic_stream ===
    # C:\Users\cosogi\chatterer\.venv\Lib\site-packages\langchain_openai\chat_models\base.py:1390: UserWarning: Cannot use method='json_schema' with model gpt-3.5-turbo since it doesn't support OpenAI's Structured Output API. You can see supported models here: https://platform.openai.com/docs/guides/structured-outputs#supported-models. To fix this warning, set `method='function_calling'. Overriding to method='function_calling'.
    # warnings.warn(
    # Async Pydantic Chunk 0: proposition_topic='The Meaning of Life' proposition_content=''
    # Async Pydantic Chunk 1: proposition_topic='The Meaning of Life' proposition_content='The'
    # Async Pydantic Chunk 2: proposition_topic='The Meaning of Life' proposition_content='The meaning'
    # Async Pydantic Chunk 3: proposition_topic='The Meaning of Life' proposition_content='The meaning of'
    # Async Pydantic Chunk 4: proposition_topic='The Meaning of Life' proposition_content='The meaning of life'
    # Async Pydantic Chunk 5: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is'
    # Async Pydantic Chunk 6: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a'
    # Async Pydantic Chunk 7: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical'
    # Async Pydantic Chunk 8: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question'
    # Async Pydantic Chunk 9: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that'
    # Async Pydantic Chunk 10: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks'
    # Async Pydantic Chunk 11: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to'
    # Async Pydantic Chunk 12: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand'
    # Async Pydantic Chunk 13: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the'
    # Async Pydantic Chunk 14: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose'
    # Async Pydantic Chunk 15: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and'
    # Async Pydantic Chunk 16: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance'
    # Async Pydantic Chunk 17: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of'
    # Async Pydantic Chunk 18: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human'
    # Async Pydantic Chunk 19: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence'
    # Async Pydantic Chunk 20: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence.'
    # Async Pydantic Chunk 21: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It'
    # Async Pydantic Chunk 22: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It is'
    # Async Pydantic Chunk 23: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It is a'
    # Async Pydantic Chunk 24: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It is a profound'
    # Async Pydantic Chunk 25: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It is a profound and'
    # Async Pydantic Chunk 26: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It is a profound and complex'
    # Async Pydantic Chunk 27: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It is a profound and complex topic'
    # Async Pydantic Chunk 28: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It is a profound and complex topic that'
    # Async Pydantic Chunk 29: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It is a profound and complex topic that has'
    # Async Pydantic Chunk 30: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It is a profound and complex topic that has been'
    # Async Pydantic Chunk 31: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It is a profound and complex topic that has been debated'
    # Async Pydantic Chunk 32: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It is a profound and complex topic that has been debated and'
    # Async Pydantic Chunk 33: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It is a profound and complex topic that has been debated and explored'
    # Async Pydantic Chunk 34: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It is a profound and complex topic that has been debated and explored by'
    # Async Pydantic Chunk 35: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It is a profound and complex topic that has been debated and explored by philosophers'
    # Async Pydantic Chunk 36: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It is a profound and complex topic that has been debated and explored by philosophers,'
    # Async Pydantic Chunk 37: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It is a profound and complex topic that has been debated and explored by philosophers, theolog'
    # Async Pydantic Chunk 38: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It is a profound and complex topic that has been debated and explored by philosophers, theologians'
    # Async Pydantic Chunk 39: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It is a profound and complex topic that has been debated and explored by philosophers, theologians,'
    # Async Pydantic Chunk 40: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It is a profound and complex topic that has been debated and explored by philosophers, theologians, and'
    # Async Pydantic Chunk 41: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It is a profound and complex topic that has been debated and explored by philosophers, theologians, and thinkers'
    # Async Pydantic Chunk 42: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It is a profound and complex topic that has been debated and explored by philosophers, theologians, and thinkers throughout'
    # Async Pydantic Chunk 43: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It is a profound and complex topic that has been debated and explored by philosophers, theologians, and thinkers throughout history'
    # Async Pydantic Chunk 44: proposition_topic='The Meaning of Life' proposition_content='The meaning of life is a philosophical question that seeks to understand the purpose and significance of human existence. It is a profound and complex topic that has been debated and explored by philosophers, theologians, and thinkers throughout history.'
