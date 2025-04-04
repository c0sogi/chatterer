"""
Integration Tests for Chatterer-based Agent System

This module contains integration tests for the complete agent system, testing all five core functionalities:
1. Tool-use
2. Delegation
3. Shared Memory
4. Context Management
5. Metacognition

These tests validate that all components work together properly in various scenarios.
"""

import asyncio
import json
import time
from typing import Any, Callable, Dict, Optional

from chatterer.language_model import Chatterer
from dotenv import load_dotenv

from .context_management import ContextAwareAgent
from .delegation import DelegatingAgent
from .metacognition import MetacognitiveAgent
from .shared_memory import MemoryAccessLevel, SharedMemory, SharedMemoryAgent
from .tool_use import ToolUsingAgent

# Load environment variables
load_dotenv()


# Test utilities
class TestResult:
    """Class to store test results"""

    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.success = False
        self.error: Optional[str] = None
        self.details: Dict[str, Any] = {}

    def complete(self, success: bool, error: Optional[str] = None, **details: Any) -> None:
        """Complete the test with results"""
        self.end_time = time.time()
        self.success = success
        self.error = error
        self.details.update(details)

    def duration(self) -> float:
        """Get test duration in seconds"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert test result to dictionary"""
        return {
            "test_name": self.test_name,
            "success": self.success,
            "duration": self.duration(),
            "error": self.error,
            "details": self.details,
        }

    def __str__(self) -> str:
        """String representation of test result"""
        status = "PASSED" if self.success else "FAILED"
        result = f"Test: {self.test_name} - {status} ({self.duration():.2f}s)"
        if self.error:
            result += f"\nError: {self.error}"
        if self.details:
            result += f"\nDetails: {json.dumps(self.details, indent=2)}"
        return result


async def run_test(test_func: Callable[..., Any], *args: Any, **kwargs: Any) -> TestResult:
    """Run a test function and return the result"""
    test_name = test_func.__name__
    result = TestResult(test_name)

    print(f"Running test: {test_name}...")

    try:
        details = await test_func(*args, **kwargs)
        result.complete(True, None, **details)
        print(f"✅ {test_name} - PASSED ({result.duration():.2f}s)")
    except Exception as e:
        result.complete(False, str(e))
        print(f"❌ {test_name} - FAILED: {str(e)} ({result.duration():.2f}s)")

    return result


# Sample tools for testing
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers together"""
    return a * b


def search_web(query: str) -> str:
    """Simulate web search"""
    return f"Results for '{query}': Example search result 1, Example search result 2"


def analyze_data(data: str) -> Dict[str, Any]:
    """Simulate data analysis"""
    return {"summary": f"Analysis of {data}", "insights": ["Insight 1", "Insight 2"]}


def fetch_weather(location: str) -> Dict[str, Any]:
    """Simulate weather fetching"""
    return {
        "location": location,
        "temperature": 22,
        "conditions": "Sunny",
        "forecast": ["Sunny", "Partly Cloudy", "Rainy"],
    }


# Test scenarios
async def test_tool_use_basic() -> Dict[str, Any]:
    """Test basic tool use functionality"""
    model = Chatterer.openai()
    agent = ToolUsingAgent(model)

    # Register tools
    agent.register_function_as_tool(add)
    agent.register_function_as_tool(multiply)

    # Execute tools directly
    add_result = await agent.execute_tool("add", a=5, b=7)
    mult_result = await agent.execute_tool("multiply", a=5, b=7)

    # Verify results
    assert add_result.success, "Add tool execution failed"
    assert mult_result.success, "Multiply tool execution failed"
    assert add_result.result == 12, f"Expected 12, got {add_result.result}"
    assert mult_result.result == 35, f"Expected 35, got {mult_result.result}"

    # Test agent run with tool use
    response = await agent.run("What is 5 + 7 and 5 * 7?")

    return {"add_result": add_result.model_dump(), "mult_result": mult_result.model_dump(), "agent_response": response}


async def test_delegation_basic() -> Dict[str, Any]:
    """Test basic delegation functionality"""
    model = Chatterer.openai()
    root_agent = DelegatingAgent(model, agent_id="root_agent")

    # Register tools
    root_agent.register_function_as_tool(add)
    root_agent.register_function_as_tool(search_web)

    # Create a sub-agent directly
    sub_agent_id = await root_agent.delegate_task(
        task_description="Calculate 10 + 20 and search for 'Python programming'",
        completion_criteria="Both calculations and search must be completed",
        relevant_information={"important_note": "This is a test task"},
    )

    # Wait for task to complete (in a real system, you'd use a proper waiting mechanism)
    await asyncio.sleep(2)

    # Get task results
    task_results = root_agent.get_all_task_results()

    # Verify results
    assert len(task_results) > 0, "No task results found"
    assert sub_agent_id in root_agent.sub_agents, f"Sub-agent {sub_agent_id} not found"

    return {"sub_agent_id": sub_agent_id, "task_results": {k: v.model_dump() for k, v in task_results.items()}}


async def test_shared_memory_basic() -> Dict[str, Any]:
    """Test basic shared memory functionality"""
    model = Chatterer.openai()
    shared_memory = SharedMemory()

    # Create agents with different memory access
    root_agent = SharedMemoryAgent(
        model=model, agent_id="root_agent", shared_memory=shared_memory, memory_access=MemoryAccessLevel.READ_WRITE
    )

    read_only_agent = SharedMemoryAgent(
        model=model, agent_id="read_only_agent", shared_memory=shared_memory, memory_access=MemoryAccessLevel.READ_ONLY
    )

    # Register tools
    root_agent.register_function_as_tool(add)
    read_only_agent.register_function_as_tool(multiply)

    # Write to memory with root agent
    write_result = await root_agent.write_to_memory(key="test_value", value=42, metadata={"description": "Test value"})

    # Read from memory with both agents
    root_read = await root_agent.read_from_memory(key="test_value")
    read_only_read = await read_only_agent.read_from_memory(key="test_value")

    # Try to write with read-only agent (should fail)
    try:
        await read_only_agent.write_to_memory(key="another_value", value="test")
        write_protection_works = False
    except Exception:
        write_protection_works = True

    # Verify results
    assert write_result is True, "Write to memory failed"
    assert root_read == 42, f"Expected 42, got {root_read}"
    assert read_only_read == 42, f"Expected 42, got {read_only_read}"
    assert write_protection_works, "Write protection failed"

    return {
        "write_result": write_result,
        "root_read": root_read,
        "read_only_read": read_only_read,
        "write_protection_works": write_protection_works,
        "memory_summary": shared_memory.get_memory_summary(),
    }


async def test_context_management_basic() -> Dict[str, Any]:
    """Test basic context management functionality"""
    model = Chatterer.openai()
    agent = ContextAwareAgent(model=model, agent_id="context_agent", max_tokens=4000, warning_threshold=0.7)

    # Register tools
    agent.register_function_as_tool(add)
    agent.register_function_as_tool(search_web)

    # Get initial token usage
    initial_usage = await agent.get_token_usage()

    # Run a simple task
    response = await agent.run("Calculate 5 + 7 and search for 'AI agents'")

    # Get updated token usage
    updated_usage = await agent.get_token_usage()

    # Compress context
    compression_result = await agent.compress_context()

    # Get final token usage
    final_usage = await agent.get_token_usage()

    # Verify results
    assert updated_usage["current_usage"]["total_tokens"] > initial_usage["current_usage"]["total_tokens"], (
        "Token usage did not increase"
    )

    return {
        "initial_usage": initial_usage,
        "updated_usage": updated_usage,
        "compression_result": compression_result,
        "final_usage": final_usage,
        "response": response,
    }


async def test_metacognition_basic() -> Dict[str, Any]:
    """Test basic metacognition functionality"""
    model = Chatterer.openai()
    agent = MetacognitiveAgent(
        model=model,
        agent_id="metacognitive_agent",
        max_tokens=4000,
        warning_threshold=0.7,
        metacognition_enabled=True,
        assessment_interval=60.0,
    )

    # Register tools
    agent.register_function_as_tool(add)
    agent.register_function_as_tool(search_web)
    agent.register_function_as_tool(analyze_data)

    # Get initial performance metrics
    initial_metrics = await agent.get_performance_metrics()

    # Create a task plan
    task_plan = await agent.create_task_plan(task_description="Calculate 10 + 20 and analyze the result", priority=2)

    # Perform self-assessment
    assessment = await agent.perform_self_assessment()

    # Get strategy adjustment
    strategy = await agent.get_strategy_adjustment()

    # Run a simple task
    response = await agent.run("Calculate 15 + 25 and perform a self-assessment")

    # Get final performance metrics
    final_metrics = await agent.get_performance_metrics()

    # Verify results
    assert task_plan["task_id"], "Task plan creation failed"
    assert assessment["assessment"]["strengths"], "Self-assessment failed"

    return {
        "initial_metrics": initial_metrics,
        "task_plan": task_plan,
        "assessment": assessment,
        "strategy": strategy,
        "final_metrics": final_metrics,
        "response": response,
    }


async def test_integrated_workflow() -> Dict[str, Any]:
    """Test a complete integrated workflow using all features"""
    model = Chatterer.openai()
    shared_memory = SharedMemory()

    # Create root agent with all capabilities
    root_agent = MetacognitiveAgent(
        model=model,
        agent_id="root_agent",
        shared_memory=shared_memory,
        memory_access=MemoryAccessLevel.READ_WRITE,
        max_tokens=6000,
        warning_threshold=0.7,
        metacognition_enabled=True,
        assessment_interval=60.0,
    )

    # Register tools
    root_agent.register_function_as_tool(add)
    root_agent.register_function_as_tool(multiply)
    root_agent.register_function_as_tool(search_web)
    root_agent.register_function_as_tool(analyze_data)
    root_agent.register_function_as_tool(fetch_weather)

    # Run a complex task that should exercise all capabilities
    prompt = """
    I need you to complete a multi-step task that will test all your capabilities:
    
    1. First, calculate 23 + 45 and store the result in shared memory as 'sum_result'
    2. Then, calculate 7 * 8 and store the result in shared memory as 'product_result'
    3. Next, delegate a task to search for information about 'artificial intelligence trends'
       with READ_ONLY access to shared memory
    4. After that, check your token usage and compress context if needed
    5. Perform a self-assessment to evaluate your approach
    6. Create a task plan for analyzing weather data
    7. Fetch weather for 'New York' and analyze the data
    8. Finally, provide a summary of all the results, including data from shared memory
    
    Throughout this process, monitor your performance and adjust your strategy if needed.
    """

    # Execute the complex workflow
    response = await root_agent.run(prompt)

    # Get memory contents
    memory_summary = shared_memory.get_memory_summary()

    # Get performance metrics
    metrics = await root_agent.get_performance_metrics()

    # Get task results
    task_results = root_agent.get_all_task_results()

    # Get final self-assessment
    assessment = await root_agent.perform_self_assessment()

    return {
        "response": response,
        "memory_summary": memory_summary,
        "performance_metrics": metrics,
        "task_results": {k: v.model_dump() for k, v in task_results.items()},
        "final_assessment": assessment,
    }


async def test_error_handling_and_recovery() -> Dict[str, Any]:
    """Test error handling and recovery capabilities"""
    model = Chatterer.openai()

    # Create agent with metacognitive capabilities
    agent = MetacognitiveAgent(
        model=model, agent_id="error_test_agent", max_tokens=4000, warning_threshold=0.7, metacognition_enabled=True
    )

    # Register tools including one that will fail
    agent.register_function_as_tool(add)

    # Register a tool that will fail
    def failing_tool(input: str) -> str:
        """A tool that will fail when called with 'fail' as input"""
        if input.lower() == "fail":
            raise ValueError("Tool failure was requested")
        return f"Processed: {input}"

    agent.register_function_as_tool(failing_tool)

    # Run a task that will trigger the error and require recovery
    prompt = """
    I need you to complete a task that will test your error handling:
    
    1. First, calculate 10 + 20 using the add tool
    2. Then, use the failing_tool with input 'fail' (this will cause an error)
    3. Detect the error and adjust your strategy
    4. Try again with the failing_tool but use 'success' as input
    5. Summarize what happened and how you recovered
    
    Show your metacognitive capabilities by explaining how you detected and handled the error.
    """

    # Execute the workflow
    response = await agent.run(prompt)

    # Get performance metrics
    metrics = await agent.get_performance_metrics()

    # Get error counts
    error_counts = metrics.get("error_counts", {})

    # Get strategy adjustment
    strategy = await agent.get_strategy_adjustment()

    return {
        "response": response,
        "performance_metrics": metrics,
        "error_counts": error_counts,
        "strategy_adjustment": strategy,
    }


async def test_token_limit_handling() -> Dict[str, Any]:
    """Test handling of token limits and context compression"""
    model = Chatterer.openai()

    # Create agent with low token limits to force compression
    agent = ContextAwareAgent(
        model=model,
        agent_id="token_test_agent",
        max_tokens=2000,  # Low limit to force compression
        warning_threshold=0.5,  # Low threshold to trigger warnings early
    )

    # Register tools
    agent.register_function_as_tool(add)
    agent.register_function_as_tool(search_web)

    # Run a verbose task that will consume tokens
    prompt = """
    I need you to complete a task that will consume many tokens:
    
    1. First, check your token usage
    2. Then, search for information about 'the history of artificial intelligence'
    3. Next, search for information about 'machine learning algorithms'
    4. Check your token usage again and compress context if needed
    5. Search for information about 'neural networks'
    6. Finally, provide a detailed summary of all the information you found
    
    Throughout this process, monitor your token usage and take appropriate actions.
    """

    # Execute the workflow
    response = await agent.run(prompt)

    # Get token usage after task
    token_usage = await agent.get_token_usage()

    # Verify compression occurred
    assert token_usage["is_approaching_limit"], "Token limit warning was not triggered"

    return {"response": response, "token_usage": token_usage}


# Main test execution
async def run_all_tests() -> Dict[str, TestResult]:
    """Run all integration tests"""

    tests = [
        test_tool_use_basic,
        test_delegation_basic,
        test_shared_memory_basic,
        test_context_management_basic,
        test_metacognition_basic,
        test_error_handling_and_recovery,
        test_token_limit_handling,
        test_integrated_workflow,
    ]

    results: Dict[str, TestResult] = {}

    for test in tests:
        result = await run_test(test)
        results[test.__name__] = result

    # Summarize results
    total = len(tests)
    passed = sum(1 for r in results.values() if r.success)

    print("\n===== TEST SUMMARY =====")
    print(f"Passed: {passed}/{total} ({(passed / total) * 100:.1f}%)")

    for name, result in results.items():
        status = "✅ PASSED" if result.success else "❌ FAILED"
        print(f"{name}: {status} ({result.duration():.2f}s)")
        if not result.success and result.error:
            print(f"  Error: {result.error}")

    return results


if __name__ == "__main__":
    # Run tests
    asyncio.run(run_all_tests())
