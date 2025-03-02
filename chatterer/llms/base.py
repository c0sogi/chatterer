from abc import ABC, abstractmethod
from typing import (
    Any,
    ClassVar,
    Iterator,
    Sequence,
    Type,
    TypeVar,
)

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field

P = TypeVar("P", bound=BaseModel)


class LLM(BaseModel, ABC):
    call_kwargs: dict[str, Any] = Field(default_factory=dict)
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    def __call__(self, messages: Sequence[ChatCompletionMessageParam]) -> str:
        return self.generate(messages)

    @abstractmethod
    def generate(self, messages: Sequence[ChatCompletionMessageParam]) -> str: ...

    @abstractmethod
    def generate_stream(self, messages: Sequence[ChatCompletionMessageParam]) -> Iterator[str]: ...

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
