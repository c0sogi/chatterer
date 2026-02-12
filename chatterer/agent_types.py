"""Type definitions for agent-based tool-use execution."""

from typing import Any, Generic, Literal, TypeVar

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

# Type variable for generic AgentResult
T = TypeVar("T")


class ToolCall(BaseModel):
    """Record of a single tool call made by the agent."""

    name: str = Field(description="Name of the tool that was called")
    args: dict[str, object] = Field(description="Arguments passed to the tool")
    result: list[dict[str, object]] = Field(description="Result returned by the tool")
    call_id: str = Field(description="Unique identifier for this tool call")


class AgentResult(BaseModel, Generic[T]):
    """
    Result from agent execution with full history.

    Generic over the type of final_answer:
    - AgentResult[str]: When no response_model is provided (default)
    - AgentResult[MyModel]: When response_model=MyModel is provided
    """

    final_answer: T = Field(description="The final answer from the agent")  # type: ignore[assignment]
    messages: list[BaseMessage] = Field(
        default_factory=list[BaseMessage],
        description="Full conversation history including tool calls",
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list[ToolCall],
        description="List of all tool invocations made during execution",
    )
    iterations: int = Field(
        default=0,
        description="Number of agent loop iterations",
    )

    model_config = {"arbitrary_types_allowed": True}


class AgentEvent(BaseModel):
    """Streaming event from agent execution."""

    type: Literal["tool_call", "tool_result", "thinking", "final_answer"] = Field(description="Type of the event")
    content: list[dict[str, object]] = Field(description="Content of the event")
    tool_name: str | None = Field(default=None, description="Name of the tool (for tool events)")
    tool_call_id: str | None = Field(default=None, description="ID of the tool call")
    args: dict[str, Any] | None = Field(default=None, description="Tool arguments (for tool_call events)")
