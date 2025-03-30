"""
Tool Use Implementation for Chatterer-based Agent System

This module implements the tool-use functionality for agents, allowing them to:
1. Use external tools provided by users
2. Execute Python code as tools
3. Store and retrieve tool execution results
"""

import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, Field, create_model

from ..language_model import Chatterer

# Type definitions
ToolFunc = Callable[..., Any]
T = TypeVar("T")


class ToolResult(BaseModel):
    """Model to store tool execution results"""

    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None


class Tool(BaseModel):
    """Model to define a tool that can be used by an agent"""

    name: str
    description: str
    function: ToolFunc
    parameters_schema: Optional[Type[BaseModel]] = None

    @property
    def param_schema(self) -> Type[BaseModel]:
        """Lazy load the parameters schema"""
        if self.parameters_schema is None:
            self.parameters_schema = self._create_parameters_schema()
        return self.parameters_schema

    def _create_parameters_schema(self) -> Type[BaseModel]:
        """Create a Pydantic model from the function signature"""
        sig = inspect.signature(self.function)
        fields: dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            # Skip self parameter for methods
            if param_name == "self":
                continue

            # Get annotation or default to Any
            annotation = param.annotation if param.annotation != inspect.Parameter.empty else Any

            # Get default value if available
            default = ... if param.default == inspect.Parameter.empty else param.default

            # Add field to schema
            fields[param_name] = (annotation, Field(default=default))

        # Create and return the model
        return create_model(f"{self.name}Parameters", **fields)

    async def execute(self, **kwargs: object) -> ToolResult:
        """Execute the tool with the provided parameters"""
        try:
            # Validate parameters against schema
            params = self.param_schema(**kwargs)

            # Get parameter values as dict
            param_dict = params.model_dump()

            # Execute function
            if inspect.iscoroutinefunction(self.function):
                result = await self.function(**param_dict)
            else:
                result = self.function(**param_dict)

            return ToolResult(tool_name=self.name, success=True, result=result)
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, result=None, error=str(e))


class ToolUsingAgent:
    """Agent that can use tools to perform tasks"""

    def __init__(self, model: Chatterer, tools: Optional[List[Tool]] = None):
        self.model = model
        self.tools: Dict[str, Tool] = {}
        self.tool_results: List[ToolResult] = []

        # Register tools if provided
        if tools:
            for tool in tools:
                self.register_tool(tool)

    def register_tool(self, tool: Tool) -> None:
        """Register a tool for the agent to use"""
        self.tools[tool.name] = tool

    def register_function_as_tool(self, func: ToolFunc, name: Optional[str] = None, description: Optional[str] = None) -> None:
        """Register a function as a tool"""
        # Use function name if name not provided
        actual_name = name if name is not None else func.__name__

        # Use function docstring if description not provided
        actual_description = description if description is not None else (func.__doc__ or f"Tool for {actual_name}")

        # Create and register tool
        tool = Tool(name=actual_name, description=actual_description, function=func)
        self.register_tool(tool)

    async def execute_tool(self, tool_name: str, **kwargs: object) -> ToolResult:
        """Execute a tool by name with the provided parameters"""
        if tool_name not in self.tools:
            return ToolResult(tool_name=tool_name, success=False, result=None, error=f"Tool '{tool_name}' not found")

        tool = self.tools[tool_name]
        result = await tool.execute(**kwargs)

        # Store result in memory
        self.tool_results.append(result)

        return result

    def get_tool_results(self, tool_name: Optional[str] = None) -> List[ToolResult]:
        """Get results of previous tool executions, optionally filtered by tool name"""
        if tool_name:
            return [r for r in self.tool_results if r.tool_name == tool_name]
        return self.tool_results

    def get_tool_descriptions(self) -> str:
        """Get descriptions of all registered tools"""
        descriptions: List[str] = []
        for name, tool in self.tools.items():
            param_info = ""
            for field_name, field in tool.param_schema.model_fields.items():
                field_type = "Any"
                if field.annotation is not None:
                    field_type = field.annotation.__name__ if hasattr(field.annotation, "__name__") else str(field.annotation)
                param_info += f"\n  - {field_name}: {field_type}"

            descriptions.append(f"Tool: {name}\nDescription: {tool.description}\nParameters:{param_info}\n")

        return "\n".join(descriptions)

    async def run(self, prompt: str) -> str:
        """Run the agent with a prompt, allowing it to use tools"""
        # Create system message with tool descriptions
        system_message = f"""You are an AI assistant that can use tools to help complete tasks.
Available tools:

{self.get_tool_descriptions()}

To use a tool, respond with:
```tool
{{"tool_name": "<tool_name>", "parameters": {{"param1": "value1", "param2": "value2"}}}}
```

After using a tool, you'll receive the result and can use it to continue your task.
You can use multiple tools by responding with multiple tool blocks.
When you're done, respond with your final answer without any tool blocks.
"""

        # Initialize conversation
        conversation = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]

        # Maximum number of tool use iterations to prevent infinite loops
        max_iterations = 10
        iterations = 0
        response = ""  # Initialize response variable

        while iterations < max_iterations:
            iterations += 1

            # Get response from model
            response = await self.model.agenerate(conversation)

            # Check if response contains tool use
            if "```tool" in response:
                # Extract tool use blocks
                tool_blocks = response.split("```tool")

                # Process each tool block
                for block in tool_blocks[1:]:
                    # Extract JSON part
                    json_str = block.split("```")[0].strip()

                    try:
                        import json

                        tool_call = json.loads(json_str)
                        tool_name = tool_call.get("tool_name")
                        parameters = tool_call.get("parameters", {})

                        # Execute tool
                        tool_result = await self.execute_tool(tool_name, **parameters)

                        # Add tool result to conversation
                        conversation.append({"role": "assistant", "content": response})
                        conversation.append({
                            "role": "system",
                            "content": f"Tool result: {tool_result.model_dump_json(indent=2)}",
                        })

                    except Exception as e:
                        # Add error to conversation
                        conversation.append({"role": "assistant", "content": response})
                        conversation.append({"role": "system", "content": f"Error parsing tool call: {str(e)}"})
            else:
                # No tool use, return final response
                return response

        # If we reach here, we've hit the maximum number of iterations
        return f"Reached maximum number of tool use iterations ({max_iterations}). Final response: {response}"


# Example usage
async def example():
    # Define some example tools
    def add(a: int, b: int) -> int:
        """Add two numbers together"""
        return a + b

    def search_web(query: str) -> str:
        """Search the web for information"""
        # This would be a real web search in a production system
        return f"Results for '{query}': Example search result 1, Example search result 2"

    # Create an agent with OpenAI model
    from chatterer.language_model import Chatterer

    model = Chatterer.openai()
    agent = ToolUsingAgent(model)

    # Register tools
    agent.register_function_as_tool(add)
    agent.register_function_as_tool(search_web)

    # Run the agent
    result = await agent.run("What is 25 + 17? Also, find information about climate change.")
    print(result)


if __name__ == "__main__":
    asyncio.run(example())
