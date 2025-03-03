from abc import ABC, abstractmethod

from ..language_model import LanguageModelInput


class BaseStrategy(ABC):
    @abstractmethod
    def invoke(self, messages: LanguageModelInput) -> str:
        """
        Invoke the strategy with the given messages.

        messages: List of messages to be passed to the strategy.
            e.g. [{"role": "user", "content": "What is the meaning of life?"}]
        """
