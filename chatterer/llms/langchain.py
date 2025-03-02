from typing import (
    Iterator,
    Sequence,
    Type,
)

from langchain_community.adapters.openai import (
    convert_openai_messages,
)
from langchain_core.language_models.chat_models import BaseChatModel
from openai.types.chat import ChatCompletionMessageParam

from .base import LLM, P


class LangchainLLM(LLM):
    client: BaseChatModel

    def generate(self, messages: Sequence[ChatCompletionMessageParam]) -> str:
        content = self.client.invoke(convert_openai_messages([dict(msg) for msg in messages])).content
        if isinstance(content, str):
            return content
        else:
            return "".join(part for part in content if isinstance(part, str))

    def generate_stream(self, messages: Sequence[ChatCompletionMessageParam]) -> Iterator[str]:
        for chunk in self.client.stream(convert_openai_messages([dict(msg) for msg in messages])):
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
        result = self.client.with_structured_output(response_model).invoke(convert_openai_messages([dict(msg) for msg in messages]))
        if isinstance(result, response_model):
            return result
        else:
            return response_model.model_validate(result)
