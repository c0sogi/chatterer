from os import environ
from typing import (
    Any,
    Iterator,
    Self,
    Sequence,
    Type,
)

from instructor import Instructor, Mode, from_openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from .base import LLM, P


class Response(BaseModel):
    response: str


class InstructorLLM(LLM):
    inst: Instructor

    @classmethod
    def openai(cls, call_kwargs: dict[str, Any] = {"model": "o3-mini"}) -> Self:
        return cls(
            inst=from_openai(OpenAI(), Mode.TOOLS_STRICT),
            call_kwargs=call_kwargs,
        )

    @classmethod
    def anthropic(
        cls,
        call_kwargs: dict[str, Any] = {
            "model": "claude-3-7-sonnet-20250219",
            "temperature": 0.7,
            "max_tokens": 8192,
        },
    ) -> Self:
        from anthropic import Anthropic
        from instructor import from_anthropic

        return cls(
            inst=from_anthropic(client=Anthropic(), mode=Mode.ANTHROPIC_TOOLS),
            call_kwargs=call_kwargs,
        )

    @classmethod
    def gemini(
        cls,
        model_name: str = "gemini-2.0-flash",
        call_kwargs: dict[str, Any] = {},
    ) -> Self:
        from google.generativeai.generative_models import GenerativeModel
        from instructor import from_gemini

        return cls(
            inst=from_gemini(client=GenerativeModel(model_name=model_name), mode=Mode.GEMINI_TOOLS),
            call_kwargs=call_kwargs,
        )

    @classmethod
    def deepseek(cls, call_kwargs: dict[str, Any] = {"model": "deepseek-chat"}) -> Self:
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

    def generate(self, messages: Sequence[ChatCompletionMessageParam]) -> str:
        if self.inst is None:
            raise ValueError("Instructor instance is not initialized")
        res = self.inst.chat.completions.create(
            response_model=Response,
            messages=list(messages),
            **self.call_kwargs,
        )
        return res.response

    def generate_stream(self, messages: Sequence[ChatCompletionMessageParam]) -> Iterator[str]:
        if self.inst is None:
            raise ValueError("Instructor instance is not initialized")
        last_content: str = ""
        for res in self.inst.chat.completions.create_partial(
            response_model=Response,
            messages=list(messages),
            **self.call_kwargs,
        ):
            content: str = res.response
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
        if self.inst is None:
            raise ValueError("Instructor instance is not initialized")
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
        if self.inst is None:
            raise ValueError("Instructor instance is not initialized")
        for res in self.inst.chat.completions.create_partial(
            response_model=response_model,
            messages=list(messages),
            **self.call_kwargs,
        ):
            yield res
