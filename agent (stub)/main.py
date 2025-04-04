"""
Main entry point for the Chatterer-based Agent System

This module provides a unified interface to create and run agents with all five core functionalities:
1. Tool-use
2. Delegation
3. Shared Memory
4. Context Management
5. Metacognition
"""

import asyncio
from typing import Any, Callable, Dict, Optional

from .metacognition import MetacognitiveAgent
from .shared_memory import MemoryAccessLevel, SharedMemory
from chatterer.language_model import Chatterer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AgentSystem:
    """Main class for creating and managing agents with all capabilities"""

    def __init__(self):
        """Initialize the agent system"""
        # Create shared memory
        self.shared_memory = SharedMemory()

        # Create OpenAI model
        self.model = Chatterer.openai()

        # Root agent with all capabilities
        self.root_agent = MetacognitiveAgent(
            model=self.model,
            agent_id="root_agent",
            shared_memory=self.shared_memory,
            memory_access=MemoryAccessLevel.READ_WRITE,
            max_tokens=8000,
            warning_threshold=0.8,
            metacognition_enabled=True,
            assessment_interval=300.0,
        )

    def register_tool(self, func: Callable[..., Any], name: Optional[str] = None, description: Optional[str] = None) -> None:
        """Register a tool with the root agent"""
        self.root_agent.register_function_as_tool(func, name, description)

    async def run(self, prompt: str) -> str:
        """Run the agent system with a prompt"""
        return await self.root_agent.run(prompt)

    def get_memory_summary(self) -> str:
        """Get a summary of shared memory contents"""
        return self.shared_memory.get_memory_summary()

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from the root agent"""
        return await self.root_agent.get_performance_metrics()

    async def perform_self_assessment(self) -> Dict[str, Any]:
        """Perform self-assessment on the root agent"""
        return await self.root_agent.perform_self_assessment()


# Example usage
async def main():
    # Create agent system
    system = AgentSystem()

    # Register some example tools
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

    system.register_tool(add)
    system.register_tool(search_web)
    system.register_tool(analyze_data)

    # Run the system with a prompt
    prompt = """
    I need you to demonstrate your capabilities:
    
    1. Calculate 25 + 17 and store the result in shared memory
    2. Search for information about 'artificial intelligence'
    3. Delegate a task to analyze the search results
    4. Check your token usage and performance metrics
    5. Perform a self-assessment
    6. Provide a summary of what you've learned and accomplished
    """

    print("Running agent system...")
    response = await system.run(prompt)

    print("\nAgent Response:")
    print(response)

    print("\nShared Memory:")
    print(system.get_memory_summary())

    print("\nPerformance Metrics:")
    metrics = await system.get_performance_metrics()
    print(metrics)

    print("\nSelf-Assessment:")
    assessment = await system.perform_self_assessment()
    print(assessment)


if __name__ == "__main__":
    asyncio.run(main())
