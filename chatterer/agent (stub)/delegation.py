"""
Delegation Mechanism Implementation for Chatterer-based Agent System

This module implements the delegation mechanism for agents, allowing them to:
1. Delegate tasks to other agents
2. Pass context and available tools to sub-agents
3. Receive results from sub-agents
4. Manage task completion conditions
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..language_model import Chatterer
from .tool_use import Tool, ToolUsingAgent


class TaskContext(BaseModel):
    """Model to store task context for delegation"""

    task_description: str
    relevant_information: Dict[str, Any] = Field(default_factory=dict)
    completion_criteria: str
    available_tools: List[str] = Field(default_factory=list)
    parent_agent_id: Optional[str] = None


class TaskResult(BaseModel):
    """Model to store task execution results"""

    task_id: str
    agent_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    sub_task_results: List[Any] = Field(default_factory=list)


class DelegatingAgent(ToolUsingAgent):
    """Agent that can delegate tasks to other agents"""

    def __init__(
        self,
        model: Chatterer,
        agent_id: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        delegation_enabled: bool = True,
        parent_agent: Optional["DelegatingAgent"] = None,
    ):
        super().__init__(model, tools)
        self.agent_id = agent_id if agent_id is not None else str(uuid.uuid4())
        self.delegation_enabled = delegation_enabled
        self.parent_agent = parent_agent
        self.sub_agents: Dict[str, "DelegatingAgent"] = {}
        self.task_results: Dict[str, TaskResult] = {}

        # Register delegation tool if enabled
        if delegation_enabled:
            self.register_function_as_tool(
                self.delegate_task, name="delegate_task", description="Delegate a task to a new sub-agent"
            )

    async def delegate_task(
        self,
        task_description: str,
        completion_criteria: str,
        relevant_information: Optional[Dict[str, Any]] = None,
        tool_names: Optional[List[str]] = None,
    ) -> str:
        """
        Delegate a task to a new sub-agent

        Args:
            task_description: Detailed description of the task to be performed
            completion_criteria: Clear criteria for when the task is considered complete
            relevant_information: Dictionary of information relevant to the task
            tool_names: List of tool names the sub-agent should have access to

        Returns:
            ID of the created sub-agent
        """
        if not self.delegation_enabled:
            raise ValueError("Delegation is not enabled for this agent")

        # Create task context
        context = TaskContext(
            task_description=task_description,
            completion_criteria=completion_criteria,
            relevant_information=relevant_information or {},
            available_tools=tool_names or list(self.tools.keys()),
            parent_agent_id=self.agent_id,
        )

        # Create sub-agent
        sub_agent_id = f"{self.agent_id}_sub_{len(self.sub_agents)}"
        sub_agent = DelegatingAgent(
            model=self.model,  # Use the same model
            agent_id=sub_agent_id,
            delegation_enabled=self.delegation_enabled,  # Pass down delegation capability
            parent_agent=self,
        )

        # Register available tools for sub-agent
        for tool_name in context.available_tools:
            if tool_name in self.tools:
                sub_agent.register_tool(self.tools[tool_name])

        # Store sub-agent
        self.sub_agents[sub_agent_id] = sub_agent

        # Create task ID
        task_id = f"task_{uuid.uuid4().hex[:8]}"

        # Execute task asynchronously
        asyncio.create_task(self._execute_delegated_task(sub_agent, task_id, context))

        return sub_agent_id

    async def _execute_delegated_task(self, sub_agent: "DelegatingAgent", task_id: str, context: TaskContext) -> None:
        """Execute a delegated task using a sub-agent"""
        try:
            # Create prompt for sub-agent
            prompt = self._create_delegation_prompt(context)

            # Run sub-agent
            result = await sub_agent.run(prompt)

            # Store task result
            task_result = TaskResult(
                task_id=task_id,
                agent_id=sub_agent.agent_id,
                success=True,
                result=result,
                sub_task_results=[r.model_dump() for r in sub_agent.get_tool_results()],
            )

            self.task_results[task_id] = task_result

            # Report back to parent agent if this is a sub-agent
            if self.parent_agent:
                await self.report_to_parent(task_result)

        except Exception as e:
            # Store error result
            task_result = TaskResult(
                task_id=task_id, agent_id=sub_agent.agent_id, success=False, result=None, error=str(e)
            )

            self.task_results[task_id] = task_result

            # Report back to parent agent if this is a sub-agent
            if self.parent_agent:
                await self.report_to_parent(task_result)

    def _create_delegation_prompt(self, context: TaskContext) -> str:
        """Create a prompt for the delegated task"""
        # Create a detailed prompt with all necessary context
        prompt = f"""# Delegated Task

## Task Description
{context.task_description}

## Completion Criteria
{context.completion_criteria}

## Relevant Information
"""

        # Add relevant information
        for key, value in context.relevant_information.items():
            prompt += f"- {key}: {value}\n"

        # Add instructions for reporting back
        prompt += """
## Instructions
1. Complete the task according to the description
2. Use the available tools as needed
3. Make sure your final response meets the completion criteria
4. Your response will be automatically reported back to the parent agent

Begin working on the task now.
"""

        return prompt

    async def report_to_parent(self, task_result: TaskResult) -> None:
        """Report task result back to parent agent"""
        if not self.parent_agent:
            raise ValueError("This agent has no parent to report to")

        # Create a reporting prompt
        report_prompt = f"""# Task Completion Report

## Task Result
Success: {task_result.success}

## Result Details
{task_result.result}

## Error (if any)
{task_result.error or "None"}

## Sub-task Results
{task_result.sub_task_results}
"""

        # Add this report to parent's context
        if self.parent_agent:
            # In a real implementation, this would add to the parent's conversation
            # For now, we'll just print it
            print(f"Agent {self.agent_id} reporting to parent {self.parent_agent.agent_id}:\n{report_prompt}")

    async def run(self, prompt: str) -> str:
        """Run the agent with a prompt, allowing it to use tools and delegate tasks"""
        # Create system message with tool descriptions and delegation capabilities
        system_message = f"""You are an AI assistant that can use tools and delegate tasks to other agents.
Available tools:

{self.get_tool_descriptions()}

To use a tool, respond with:
```tool
{{"tool_name": "<tool_name>", "parameters": {{"param1": "value1", "param2": "value2"}}}}
```

You can delegate complex tasks to other agents using the delegate_task tool.
When delegating, provide clear task description, completion criteria, and relevant information.

After using a tool, you'll receive the result and can use it to continue your task.
You can use multiple tools by responding with multiple tool blocks.
When you're done, respond with your final answer without any tool blocks.
"""

        # Initialize conversation
        conversation = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]

        # Maximum number of tool use iterations to prevent infinite loops
        max_iterations = 15
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

    def get_all_task_results(self) -> Dict[str, TaskResult]:
        """Get all task results from this agent and its sub-agents"""
        all_results = self.task_results.copy()

        # Add results from sub-agents
        for sub_agent_id, sub_agent in self.sub_agents.items():
            sub_results = sub_agent.get_all_task_results()
            for task_id, result in sub_results.items():
                all_results[f"{sub_agent_id}_{task_id}"] = result

        return all_results


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

    def analyze_data(data: str) -> Dict[str, Any]:
        """Analyze data and return insights"""
        # This would be a real data analysis in a production system
        return {"summary": "Data analysis summary", "insights": ["Insight 1", "Insight 2"]}

    # Create an agent with OpenAI model
    from chatterer.language_model import Chatterer

    model = Chatterer.openai()
    agent = DelegatingAgent(model, agent_id="root_agent")

    # Register tools
    agent.register_function_as_tool(add)
    agent.register_function_as_tool(search_web)
    agent.register_function_as_tool(analyze_data)

    # Run the agent
    result = await agent.run(
        "I need to solve a complex problem. First, calculate 25 + 17. "
        + "Then, delegate a task to search for information about climate change. "
        + "Finally, analyze the search results."
    )
    print(result)

    # Print all task results
    print("\nAll task results:")
    for task_id, task_result in agent.get_all_task_results().items():
        print(f"Task {task_id}: {'Success' if task_result.success else 'Failed'}")
        print(f"Result: {task_result.result}")
        print()


if __name__ == "__main__":
    asyncio.run(example())
