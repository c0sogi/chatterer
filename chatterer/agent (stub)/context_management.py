"""
Context Management Implementation for Chatterer-based Agent System

This module implements the context management functionality for agents, allowing them to:
1. Monitor token usage and prevent exceeding token limits
2. Compress or summarize context when needed
3. Manage conversation history efficiently
4. Handle graceful termination when approaching token limits
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..language_model import Chatterer
from .delegation import DelegatingAgent
from .shared_memory import MemoryAccessLevel, SharedMemory, SharedMemoryAgent
from .tool_use import Tool


class TokenUsage(BaseModel):
    """Model to track token usage"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def update(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Update token usage"""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens


class ConversationSummary(BaseModel):
    """Model to store conversation summary"""

    summary: str
    key_points: List[str] = Field(default_factory=list)
    token_count: int
    original_token_count: int
    compression_ratio: float


class ContextManager:
    """Manager for agent context and token usage"""

    def __init__(self, model: Chatterer, max_tokens: int = 8000, warning_threshold: float = 0.8):
        self.model = model
        self.max_tokens = max_tokens
        self.warning_threshold = warning_threshold
        self.token_usage = TokenUsage()
        self.conversation_history: List[Dict[str, str]] = []
        self.is_approaching_limit = False
        self.is_critical = False

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history and update token count"""
        message = {"role": role, "content": content}
        self.conversation_history.append(message)

        # Estimate token count for this message
        # This is a simple estimation; in production, you'd use a proper tokenizer
        estimated_tokens = self._estimate_tokens(content)

        # Update token usage
        if role == "assistant":
            self.token_usage.update(0, estimated_tokens)
        else:
            self.token_usage.update(estimated_tokens, 0)

        # Check if approaching token limit
        self._check_token_limit()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text (simple approximation)"""
        # A very rough approximation: 1 token ≈ 4 characters for English text
        # In production, use the actual tokenizer from the model
        return len(text) // 4 + 1

    def _check_token_limit(self) -> None:
        """Check if approaching token limit and update status flags"""
        usage_ratio = self.token_usage.total_tokens / self.max_tokens

        # Update status flags
        self.is_approaching_limit = usage_ratio >= self.warning_threshold
        self.is_critical = usage_ratio >= 0.95

    async def summarize_conversation(self, start_idx: int = 0, end_idx: Optional[int] = None) -> ConversationSummary:
        """Summarize a portion of the conversation history"""
        if end_idx is None:
            end_idx = len(self.conversation_history)

        # Extract the conversation segment to summarize
        conversation_segment = self.conversation_history[start_idx:end_idx]

        # Calculate original token count
        original_token_count = sum(self._estimate_tokens(msg["content"]) for msg in conversation_segment)

        # Create a prompt for summarization
        prompt = "Please summarize the following conversation, extracting key points and important information:\n\n"
        for msg in conversation_segment:
            prompt += f"{msg['role'].capitalize()}: {msg['content']}\n\n"

        # Add instructions for the summary format
        prompt += "\nProvide a concise summary and list key points in a structured format."

        # Generate summary using the model
        summary_response = await self.model.agenerate([
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes conversations accurately and concisely.",
            },
            {"role": "user", "content": prompt},
        ])

        # Parse the summary and key points
        # This is a simple parsing; in production, you might want to use a more structured approach
        summary_text = summary_response
        key_points = []

        # Extract key points if they're in a list format
        if "Key points:" in summary_text:
            parts = summary_text.split("Key points:")
            summary_text = parts[0].strip()
            points_text = parts[1].strip()

            # Extract points (assuming they're in a list with numbers, bullets, or dashes)
            import re

            key_points = [point.strip() for point in re.split(r"[\d*\-•]+\.?\s+", points_text) if point.strip()]

        # Calculate token count for the summary
        summary_token_count = self._estimate_tokens(summary_text) + sum(
            self._estimate_tokens(point) for point in key_points
        )

        # Calculate compression ratio
        compression_ratio = original_token_count / max(1, summary_token_count)

        return ConversationSummary(
            summary=summary_text,
            key_points=key_points,
            token_count=summary_token_count,
            original_token_count=original_token_count,
            compression_ratio=compression_ratio,
        )

    async def compress_context(self) -> Tuple[bool, str]:
        """Compress conversation context when approaching token limit"""
        if len(self.conversation_history) < 4:
            # Not enough context to compress
            return False, "Not enough context to compress"

        # Keep the system message and the most recent user and assistant messages
        system_messages = [msg for msg in self.conversation_history if msg["role"] == "system"]
        recent_messages = self.conversation_history[-4:]  # Keep last 2 exchanges (4 messages)

        # Determine the segment to summarize (everything except system messages and recent messages)
        to_summarize = [
            msg for msg in self.conversation_history if msg not in system_messages and msg not in recent_messages
        ]

        if not to_summarize:
            return False, "No messages to compress"

        # Summarize the conversation segment
        summary = await self.summarize_conversation(
            start_idx=self.conversation_history.index(to_summarize[0]),
            end_idx=self.conversation_history.index(to_summarize[-1]) + 1,
        )

        # Create a new compressed conversation history
        compressed_history = system_messages.copy()

        # Add a summary message
        summary_message = {"role": "system", "content": f"[Previous conversation summary: {summary.summary}]"}
        compressed_history.append(summary_message)

        # Add key points if available
        if summary.key_points:
            key_points_message = {
                "role": "system",
                "content": "[Key points from previous conversation: " + "; ".join(summary.key_points) + "]",
            }
            compressed_history.append(key_points_message)

        # Add recent messages
        compressed_history.extend(recent_messages)

        # Update conversation history and recalculate token usage
        old_total = self.token_usage.total_tokens
        self.conversation_history = compressed_history

        # Recalculate token usage
        self.token_usage = TokenUsage()
        for msg in self.conversation_history:
            if msg["role"] == "assistant":
                self.token_usage.update(0, self._estimate_tokens(msg["content"]))
            else:
                self.token_usage.update(self._estimate_tokens(msg["content"]), 0)

        # Check token limit again
        self._check_token_limit()

        new_total = self.token_usage.total_tokens
        savings = old_total - new_total

        return True, f"Compressed context, saving {savings} tokens. New total: {new_total} tokens."

    def get_conversation_for_model(self) -> List[Dict[str, str]]:
        """Get the conversation history in a format suitable for the model"""
        return self.conversation_history.copy()

    def get_token_usage_info(self) -> Dict[str, Any]:
        """Get information about token usage"""
        return {
            "current_usage": self.token_usage.model_dump(),
            "max_tokens": self.max_tokens,
            "usage_percentage": (self.token_usage.total_tokens / self.max_tokens) * 100,
            "is_approaching_limit": self.is_approaching_limit,
            "is_critical": self.is_critical,
            "message_count": len(self.conversation_history),
        }

    def should_delegate(self) -> bool:
        """Determine if the task should be delegated due to token usage"""
        # If we're approaching the token limit and have enough context, consider delegation
        return self.is_approaching_limit and len(self.conversation_history) > 10

    def should_terminate(self) -> bool:
        """Determine if the agent should terminate due to token usage"""
        return self.is_critical


class ContextAwareAgent(SharedMemoryAgent):
    """Agent with context management capabilities"""

    def __init__(
        self,
        model: Chatterer,
        agent_id: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        delegation_enabled: bool = True,
        parent_agent: Optional[DelegatingAgent] = None,
        shared_memory: Optional[SharedMemory] = None,
        memory_access: MemoryAccessLevel = MemoryAccessLevel.NO_ACCESS,
        max_tokens: int = 8000,
        warning_threshold: float = 0.8,
    ):
        if tools is None:
            tools = []
        if agent_id is None:
            agent_id = ""

        super().__init__(model, agent_id, tools, delegation_enabled, parent_agent, shared_memory, memory_access)

        # Initialize context manager
        self.context_manager = ContextManager(model, max_tokens, warning_threshold)

        # Register context management tools
        self.register_function_as_tool(
            self.get_token_usage, name="get_token_usage", description="Get information about current token usage"
        )

        self.register_function_as_tool(
            self.compress_context, name="compress_context", description="Compress conversation context to save tokens"
        )

    async def get_token_usage(self) -> Dict[str, Any]:
        """Get information about current token usage"""
        return self.context_manager.get_token_usage_info()

    async def compress_context(self) -> str:
        """Compress conversation context to save tokens"""
        _, message = await self.context_manager.compress_context()
        return message

    async def run(self, prompt: str) -> str:
        """Run the agent with a prompt, with context management"""
        # Create system message with tool descriptions, delegation capabilities, shared memory info, and context management
        memory_info = ""
        if hasattr(self, "shared_memory") and self.shared_memory:
            readable_keys = await self.list_memory_keys()
            if readable_keys:
                memory_info = "\nYou have access to shared memory with the following keys:\n"
                memory_info += ", ".join(readable_keys)
                memory_info += "\nUse the memory tools to read from or write to shared memory."

        context_info = """
You have context management capabilities to handle token limits efficiently.
Use the get_token_usage tool to monitor your token usage.
If you're approaching the token limit, consider:
1. Using the compress_context tool to summarize previous conversation
2. Delegating complex subtasks to other agents
3. Completing your current task and providing a final response

Always be mindful of your token usage to ensure you can complete tasks effectively.
"""

        system_message = f"""You are an AI assistant that can use tools, delegate tasks to other agents, access shared memory, and manage context efficiently.
Available tools:

{self.get_tool_descriptions()}

To use a tool, respond with:
```tool
{{"tool_name": "<tool_name>", "parameters": {{"param1": "value1", "param2": "value2"}}}}
```

You can delegate complex tasks to other agents using the delegate_task tool.
When delegating, provide clear task description, completion criteria, and relevant information.
{memory_info}
{context_info}

After using a tool, you'll receive the result and can use it to continue your task.
You can use multiple tools by responding with multiple tool blocks.
When you're done, respond with your final answer without any tool blocks.
"""

        # Initialize conversation in context manager
        self.context_manager.add_message("system", system_message)
        self.context_manager.add_message("user", prompt)

        # Maximum number of tool use iterations to prevent infinite loops
        max_iterations = 15
        iterations = 0

        while iterations < max_iterations:
            iterations += 1

            # Check if we should terminate due to token limit
            if self.context_manager.should_terminate():
                return "I've reached my token limit and need to conclude. Here's what I've accomplished so far: [summary of work done]"

            # Check if we should compress context
            if self.context_manager.is_approaching_limit and iterations > 5:
                await self.compress_context()

            # Get conversation history from context manager
            conversation = self.context_manager.get_conversation_for_model()

            # Get response from model
            response = await self.model.agenerate(conversation)

            # Add assistant response to context manager
            self.context_manager.add_message("assistant", response)

            # Check if response contains tool use
            if "```tool" in response:
                # Extract tool use blocks
                tool_blocks = response.split("```tool")

                # Process each tool block
                for block in tool_blocks[1:]:
                    # Extract JSON part
                    json_str = block.split("```")[0].strip()

                    try:
                        tool_call = json.loads(json_str)
                        tool_name = tool_call.get("tool_name")
                        parameters = tool_call.get("parameters", {})

                        # Execute tool
                        tool_result = await self.execute_tool(tool_name, **parameters)

                        # Add tool result to context manager
                        result_str = f"Tool result: {tool_result.model_dump_json(indent=2)}"
                        self.context_manager.add_message("system", result_str)

                    except Exception as e:
                        # Add error to context manager
                        error_str = f"Error parsing tool call: {str(e)}"
                        self.context_manager.add_message("system", error_str)
            else:
                # No tool use, return final response
                return response

        # If maximum iterations reached, return a message
        return "Maximum number of tool use iterations reached. Here's what I've accomplished so far: [summary of work done]"
