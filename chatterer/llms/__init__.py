from importlib.util import find_spec

from .base import LLM

__all__ = ["LLM"]

if find_spec("langchain_core") is not None:
    from .langchain import LangchainLLM

    __all__ += ["LangchainLLM"]

if find_spec("ollama") is not None:
    from .ollama import OllamaLLM

    __all__ += ["OllamaLLM"]

if find_spec("instructor") is not None:
    from .instructor import InstructorLLM

    __all__ += ["InstructorLLM"]
