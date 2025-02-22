from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import environ
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Self,
    Sequence,
    Type,
    TypeVar,
)

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, create_model
from pydantic.json_schema import JsonSchemaValue

if TYPE_CHECKING:
    from instructor import Instructor
    from langchain_core.language_models.chat_models import BaseChatModel
    from ollama import Options, Tool

P = TypeVar("P", bound=BaseModel)


@dataclass
class LLM(ABC):
    call_kwargs: dict[str, Any]

    @abstractmethod
    def generate(
        self, messages: Sequence[ChatCompletionMessageParam]
    ) -> str: ...

    @abstractmethod
    def generate_stream(
        self, messages: Sequence[ChatCompletionMessageParam]
    ) -> Iterator[str]: ...

    @abstractmethod
    def generate_pydantic(
        self,
        response_model: Type[P],
        messages: Sequence[ChatCompletionMessageParam],
    ) -> P: ...

    def generate_pydantic_stream(
        self,
        response_model: Type[P],
        messages: Sequence[ChatCompletionMessageParam],
    ) -> Iterator[P]:
        raise NotImplementedError


@dataclass
class InstructorLLM(LLM):
    inst: "Instructor"

    @property
    def dependency(self) -> list[str]:
        return ["instructor"]

    @classmethod
    def openai(
        cls, call_kwargs: dict[str, Any] = {"model": "o3-mini"}
    ) -> Self:
        from instructor import Mode, from_openai

        return cls(
            inst=from_openai(OpenAI(), Mode.TOOLS_STRICT),
            call_kwargs=call_kwargs,
        )

    @classmethod
    def anthropic(
        cls,
        call_kwargs: dict[str, Any] = {
            "temperature": 0.7,
            "max_tokens": 8192,
            "model": "claude-3-5-sonnet-20241022",
        },
    ) -> Self:

        from anthropic import Anthropic
        from instructor import Mode, from_anthropic

        return cls(
            inst=from_anthropic(
                client=Anthropic(), mode=Mode.ANTHROPIC_TOOLS
            ),
            call_kwargs=call_kwargs,
        )

    @classmethod
    def deepseek(
        cls, call_kwargs: dict[str, Any] = {"model": "deepseek-chat"}
    ) -> Self:

        from instructor import Mode, from_openai

        return cls(
            inst=from_openai(
                OpenAI(
                    base_url="https://api.deepseek.com/v1",
                    api_key=environ["DEEPSEEK_API_KEY"],
                ),
                Mode.TOOLS_STRICT,
            ),
            call_kwargs=call_kwargs,
        )

    def generate(
        self, messages: Sequence[ChatCompletionMessageParam]
    ) -> str:
        res = self.inst.chat.completions.create(
            response_model=create_model(
                "Response",
                response=(str, ...),
            ),
            messages=list(messages),
            **self.call_kwargs,
        )
        return str(getattr(res, "response", "") or "")

    def generate_stream(
        self, messages: Sequence[ChatCompletionMessageParam]
    ) -> Iterator[str]:
        last_content: str = ""
        for res in self.inst.chat.completions.create_partial(
            response_model=create_model(
                "Response",
                response=(str, ...),
            ),
            messages=list(messages),
            **self.call_kwargs,
        ):
            content = str(getattr(res, "response", "") or "")
            delta: str = content.removeprefix(last_content)
            if not delta:
                continue
            last_content = content
            yield delta

    def generate_pydantic(
        self,
        response_model: Type[P],
        messages: Sequence[ChatCompletionMessageParam],
    ) -> P:
        return self.inst.chat.completions.create(
            response_model=response_model,
            messages=list(messages),
            **self.call_kwargs,
        )

    def generate_pydantic_stream(
        self,
        response_model: Type[P],
        messages: Sequence[ChatCompletionMessageParam],
    ) -> Iterator[P]:
        for res in self.inst.chat.completions.create_partial(
            response_model=response_model,
            messages=list(messages),
            **self.call_kwargs,
        ):
            yield res


@dataclass
class OllamaLLM(LLM):
    model: str
    tools: Optional[Sequence[Mapping[str, Any] | "Tool" | Callable]] = None
    stream: bool = False
    format: Optional[Literal["", "json"] | JsonSchemaValue] = None
    options: Optional[Mapping[str, Any] | "Options"] = None
    keep_alive: Optional[float | str] = None

    def generate(
        self, messages: Sequence[ChatCompletionMessageParam]
    ) -> str:
        return "".join(self.generate_stream(messages))

    def generate_stream(
        self, messages: Sequence[ChatCompletionMessageParam]
    ) -> Iterator[str]:
        from ollama import chat

        model = str(self.call_kwargs.get("model", self.model))
        format = self.call_kwargs.get("format", self.format)
        options = self.call_kwargs.get("options", self.options)
        keep_alive = self.call_kwargs.get("keep_alive", self.keep_alive)
        tools = self.call_kwargs.get("tools", self.tools)
        return (
            res.message.content or ""
            for res in chat(
                model=model,
                messages=messages,
                tools=tools,
                stream=True,
                format=format,
                options=options,
                keep_alive=keep_alive,
            )
        )

    def generate_pydantic(
        self,
        response_model: Type[P],
        messages: Sequence[ChatCompletionMessageParam],
    ) -> P:
        from ollama import chat

        model = str(self.call_kwargs.get("model", self.model))
        format = response_model.model_json_schema()
        options = self.call_kwargs.get("options", self.options)
        keep_alive = self.call_kwargs.get("keep_alive", self.keep_alive)
        return response_model.model_validate_json(
            chat(
                model=model,
                messages=messages,
                tools=None,
                stream=False,
                format=format,
                options=options,
                keep_alive=keep_alive,
            ).message.content
            or ""
        )


@dataclass
class LangchainLLM(LLM):
    client: "BaseChatModel"

    def generate(
        self, messages: Sequence[ChatCompletionMessageParam]
    ) -> str:
        from langchain_community.adapters.openai import (
            convert_openai_messages,
        )

        content = self.client.invoke(
            convert_openai_messages([dict(msg) for msg in messages])
        ).content
        if isinstance(content, str):
            return content
        else:
            return "".join(part for part in content if isinstance(part, str))

    def generate_stream(
        self, messages: Sequence[ChatCompletionMessageParam]
    ) -> Iterator[str]:
        from langchain_community.adapters.openai import (
            convert_openai_messages,
        )

        for chunk in self.client.stream(
            convert_openai_messages([dict(msg) for msg in messages])
        ):
            content = chunk.content
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
        response_model: Type[P],
        messages: Sequence[ChatCompletionMessageParam],
    ) -> P:
        from langchain_community.adapters.openai import (
            convert_openai_messages,
        )

        result = self.client.with_structured_output(response_model).invoke(
            convert_openai_messages([dict(msg) for msg in messages])
        )
        if isinstance(result, response_model):
            return result
        else:
            return response_model.model_validate(result)
