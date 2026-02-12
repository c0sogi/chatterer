from dotenv import load_dotenv

from .agent_types import AgentEvent, AgentResult, ToolCall
from .constants import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_GOOGLE_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_XAI_MODEL,
)
from .interactive import interactive_shell
from .language_model import Chatterer
from .messages import (
    AIMessage,
    BaseMessage,
    BaseMessageChunk,
    FunctionMessage,
    HumanMessage,
    LanguageModelInput,
    SystemMessage,
    UsageMetadata,
)
from b64image import Base64Image, what
from .utils.code_agent import CodeExecutionResult, FunctionSignature
from .utils.code_snippets import CodeSnippets

load_dotenv()

__all__ = [
    # Core
    "Chatterer",
    # Agent types
    "AgentResult",
    "AgentEvent",
    "ToolCall",
    # Messages
    "BaseMessage",
    "HumanMessage",
    "SystemMessage",
    "AIMessage",
    "FunctionMessage",
    "BaseMessageChunk",
    "LanguageModelInput",
    "UsageMetadata",
    # Code execution
    "CodeSnippets",
    "Base64Image",
    "FunctionSignature",
    "CodeExecutionResult",
    # Interactive
    "interactive_shell",
    # Constants
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_GOOGLE_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_OPENROUTER_MODEL",
    "DEFAULT_XAI_MODEL",
    # Utils
    "what",
]
