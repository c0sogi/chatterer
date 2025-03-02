from typing import (
    Any,
    Callable,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Type,
)

from ollama import Options, Tool, chat
from openai.types.chat import ChatCompletionMessageParam
from pydantic.json_schema import JsonSchemaValue

from .base import LLM, P


class OllamaLLM(LLM):
    model: str
    tools: Optional[Sequence[Mapping[str, Any] | Tool | Callable]] = None
    stream: bool = False
    format: Optional[Literal["", "json"] | JsonSchemaValue] = None
    options: Optional[Mapping[str, Any] | Options] = None
    keep_alive: Optional[float | str] = None

    def generate(self, messages: Sequence[ChatCompletionMessageParam]) -> str:
        return "".join(self.generate_stream(messages))

    def generate_stream(self, messages: Sequence[ChatCompletionMessageParam]) -> Iterator[str]:
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
