"""
Metacognition Implementation for Chatterer-based Agent System

This module implements the metacognition capabilities for agents, allowing them to:
1. Monitor their own state and performance
2. Evaluate the quality of their decisions and actions
3. Adjust strategies based on self-assessment
4. Detect and recover from errors or suboptimal behavior
5. Optimize resource usage and task planning
"""

import time
import uuid
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from ..language_model import Chatterer
from .context_management import ContextAwareAgent


class AgentState(Enum):
    """Enum for agent state"""

    IDLE = auto()
    PLANNING = auto()
    EXECUTING = auto()
    EVALUATING = auto()
    DELEGATING = auto()
    REPORTING = auto()
    ERROR = auto()
    TERMINATING = auto()


class PerformanceMetrics(BaseModel):
    """Model to track agent performance metrics"""

    task_success_rate: float = 1.0
    average_task_time: float = 0.0
    tool_usage_counts: Dict[str, int] = Field(default_factory=dict)
    error_counts: Dict[str, int] = Field(default_factory=dict)
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_tokens_used: int = 0
    total_execution_time: float = 0.0

    def update_task_metrics(self, success: bool, execution_time: float, tokens_used: int) -> None:
        """Update task-related metrics"""
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1

        # Update success rate
        total_tasks = self.tasks_completed + self.tasks_failed
        self.task_success_rate = self.tasks_completed / max(1, total_tasks)

        # Update average task time
        self.total_execution_time += execution_time
        self.average_task_time = self.total_execution_time / max(1, total_tasks)

        # Update token usage
        self.total_tokens_used += tokens_used

    def record_tool_usage(self, tool_name: str) -> None:
        """Record tool usage"""
        if tool_name in self.tool_usage_counts:
            self.tool_usage_counts[tool_name] += 1
        else:
            self.tool_usage_counts[tool_name] = 1

    def record_error(self, error_type: str) -> None:
        """Record an error"""
        if error_type in self.error_counts:
            self.error_counts[error_type] += 1
        else:
            self.error_counts[error_type] = 1

    def get_most_used_tools(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Get the most frequently used tools"""
        sorted_tools = sorted(self.tool_usage_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_tools[:limit]

    def get_most_common_errors(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Get the most common errors"""
        sorted_errors = sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_errors[:limit]

    def get_summary(self) -> str:
        """Get a summary of performance metrics"""
        summary = "Performance Metrics Summary:\n"
        summary += f"- Tasks completed: {self.tasks_completed}\n"
        summary += f"- Tasks failed: {self.tasks_failed}\n"
        summary += f"- Success rate: {self.task_success_rate:.2%}\n"
        summary += f"- Average task time: {self.average_task_time:.2f} seconds\n"
        summary += f"- Total tokens used: {self.total_tokens_used}\n"

        # Add most used tools
        most_used = self.get_most_used_tools(3)
        if most_used:
            summary += "- Most used tools:\n"
            for tool, count in most_used:
                summary += f"  - {tool}: {count} times\n"

        # Add most common errors
        most_common_errors = self.get_most_common_errors(3)
        if most_common_errors:
            summary += "- Most common errors:\n"
            for error, count in most_common_errors:
                summary += f"  - {error}: {count} times\n"

        return summary


class SelfAssessment(BaseModel):
    """Model for agent self-assessment"""

    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    improvement_areas: List[str] = Field(default_factory=list)
    confidence_level: float = 0.5  # 0.0 to 1.0
    reasoning: str = ""


class TaskPlan(BaseModel):
    """Model for a task execution plan"""

    task_id: str
    steps: List[str] = Field(default_factory=list)
    estimated_tokens: int = 0
    estimated_time: float = 0.0
    dependencies: List[str] = Field(default_factory=list)
    priority: int = 1  # 1 (highest) to 5 (lowest)
    status: str = "pending"


class MetacognitiveMonitor:
    """Monitor for agent metacognition"""

    def __init__(self, model: Chatterer):
        self.model = model
        self.performance_metrics = PerformanceMetrics()
        self.state_history: List[Tuple[AgentState, float]] = []
        self.current_state: AgentState = AgentState.IDLE
        self.task_plans: Dict[str, TaskPlan] = {}
        self.last_assessment_time: float = 0.0
        self.assessment_interval: float = 300.0  # 5 minutes
        self.last_self_assessment: Optional[SelfAssessment] = None
        self.error_threshold: int = 3  # Number of errors before triggering reassessment
        self.consecutive_errors: int = 0
        self.start_time: float = time.time()

    def set_state(self, state: AgentState) -> None:
        """Set the current agent state"""
        self.current_state = state
        self.state_history.append((state, time.time()))

    def record_tool_usage(self, tool_name: str) -> None:
        """Record tool usage"""
        self.performance_metrics.record_tool_usage(tool_name)

    def record_error(self, error_type: str) -> None:
        """Record an error and update consecutive error count"""
        self.performance_metrics.record_error(error_type)
        self.consecutive_errors += 1

        # If we've hit the error threshold, set state to ERROR
        if self.consecutive_errors >= self.error_threshold:
            self.set_state(AgentState.ERROR)

    def record_success(self) -> None:
        """Record a successful operation and reset consecutive error count"""
        self.consecutive_errors = 0

    def update_task_metrics(self, task_id: str, success: bool) -> None:
        """Update task metrics when a task is completed"""
        if task_id in self.task_plans:
            plan = self.task_plans[task_id]
            execution_time = time.time() - self.start_time
            tokens_used = plan.estimated_tokens  # In a real system, get actual token usage

            self.performance_metrics.update_task_metrics(success, execution_time, tokens_used)

            # Update plan status
            plan.status = "completed" if success else "failed"

    def create_task_plan(self, task_description: str, priority: int = 3) -> str:
        """Create a task execution plan"""
        task_id = f"task_{uuid.uuid4().hex[:8]}"

        # In a real system, this would use the model to generate a plan
        # For now, we'll create a simple plan
        steps: List[str] = [
            f"Analyze task: {task_description}",
            "Identify required tools and information",
            "Execute primary task operations",
            "Evaluate results",
            "Report findings",
        ]

        # Estimate tokens based on task description length
        estimated_tokens = len(task_description) * 5

        # Estimate time based on tokens (very rough estimate)
        estimated_time = estimated_tokens / 100

        plan = TaskPlan(
            task_id=task_id,
            steps=steps,
            estimated_tokens=estimated_tokens,
            estimated_time=estimated_time,
            priority=priority,
            status="pending",
        )

        self.task_plans[task_id] = plan
        return task_id

    def should_reassess(self) -> bool:
        """Determine if the agent should perform self-assessment"""
        current_time = time.time()

        # Reassess if we've hit the error threshold
        if self.consecutive_errors >= self.error_threshold:
            return True

        # Reassess if it's been long enough since the last assessment
        if current_time - self.last_assessment_time >= self.assessment_interval:
            return True

        # Reassess if we've never done an assessment
        if self.last_self_assessment is None:
            return True

        return False

    async def perform_self_assessment(self, recent_conversation: List[Dict[str, str]]) -> SelfAssessment:
        """Perform self-assessment based on recent performance and conversation"""
        # Update last assessment time
        self.last_assessment_time = time.time()

        # Create a prompt for self-assessment
        metrics_summary = self.performance_metrics.get_summary()

        # Get state history summary
        state_summary = "Recent state transitions:\n"
        for _, (state, timestamp) in enumerate(self.state_history[-5:]):
            state_summary += f"- {state.name} at {time.ctime(timestamp)}\n"

        prompt = f"""Please perform a self-assessment based on the following information:

Performance Metrics:
{metrics_summary}

State History:
{state_summary}

Recent Conversation:
"""

        # Add recent conversation
        for msg in recent_conversation[-5:]:
            prompt += f"{msg['role'].capitalize()}: {msg['content'][:100]}...\n"

        prompt += """
Based on this information, please:
1. Identify 2-3 strengths in my performance
2. Identify 2-3 weaknesses or areas for improvement
3. Suggest 2-3 specific actions I can take to improve
4. Assess my overall confidence level (0.0 to 1.0)
5. Provide brief reasoning for this assessment

Format your response as a structured assessment with clear sections.
"""

        # Generate self-assessment using the model
        assessment_response = await self.model.agenerate([
            {
                "role": "system",
                "content": "You are a metacognitive assistant that helps AI agents evaluate their own performance.",
            },
            {"role": "user", "content": prompt},
        ])

        # Parse the assessment
        # This is a simple parsing; in production, you might want to use a more structured approach
        assessment_text = assessment_response

        # Extract strengths, weaknesses, improvements, and confidence
        strengths: List[str] = []
        weaknesses: List[str] = []
        improvements: List[str] = []
        confidence = 0.5
        reasoning = ""

        # Track current section
        current_section: Optional[str] = None
        list_sections: Dict[str, List[str]] = {
            "Strengths": strengths,
            "Weaknesses": weaknesses,
            "Improvement": improvements,
        }

        for line in assessment_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Check if this is a section header
            for section_name in ["Strengths", "Weaknesses", "Improvement", "Confidence", "Reasoning"]:
                if section_name.lower() in line.lower() and ":" in line:
                    current_section = section_name
                    if section_name == "Confidence":
                        # Try to extract confidence value
                        try:
                            confidence_str = line.split(":")[-1].strip()
                            confidence = float(confidence_str)
                        except Exception:
                            pass
                    break

            # If we're in a list section, extract items
            if current_section in list_sections and (
                line.startswith("-") or line.startswith("*") or (line[0].isdigit() and line[1] in [".", ")"])
            ):
                item = line.split(" ", 1)[1].strip()
                section_list = list_sections[current_section]
                section_list.append(item)

            # If we're in the reasoning section, collect text
            if current_section == "Reasoning" and not line.startswith("Reasoning"):
                reasoning += line + " "

        # Create and return self-assessment
        self.last_self_assessment = SelfAssessment(
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_areas=improvements,
            confidence_level=min(1.0, max(0.0, confidence)),  # Ensure confidence is between 0 and 1
            reasoning=reasoning.strip(),
        )

        return self.last_self_assessment

    def get_state_duration(self, state: AgentState) -> float:
        """Get the total time spent in a particular state"""
        duration = 0.0

        for i, (s, timestamp) in enumerate(self.state_history):
            if s == state:
                # Find the end of this state period
                end_time = time.time()
                for j in range(i + 1, len(self.state_history)):
                    if self.state_history[j][0] != state:
                        end_time = self.state_history[j][1]
                        break

                duration += end_time - timestamp

        return duration

    def get_recommendations(self) -> List[str]:
        """Get recommendations based on current state and performance"""
        recommendations: List[str] = []

        # Check if we're spending too much time in certain states
        planning_time = self.get_state_duration(AgentState.PLANNING)
        executing_time = self.get_state_duration(AgentState.EXECUTING)

        if planning_time > executing_time * 2:
            recommendations.append(
                "You're spending too much time planning. Consider simplifying your approach or delegating more tasks."
            )

        # Check error rate
        if self.performance_metrics.task_success_rate < 0.7:
            recommendations.append(
                "Your task success rate is low. Consider breaking tasks into smaller steps or using more reliable tools."
            )

        # Check token usage efficiency
        avg_tokens_per_task = self.performance_metrics.total_tokens_used / max(
            1, self.performance_metrics.tasks_completed
        )
        if avg_tokens_per_task > 2000:
            recommendations.append(
                "Your token usage per task is high. Consider compressing context more frequently or delegating subtasks."
            )

        # Add recommendations from last self-assessment
        if self.last_self_assessment:
            for area in self.last_self_assessment.improvement_areas:
                recommendations.append(f"Self-assessment suggestion: {area}")

        return recommendations

    def should_delegate_based_on_metacognition(self, task_description: str, context_size: int) -> bool:
        """Determine if a task should be delegated based on metacognitive assessment"""
        # If we have low confidence, delegate more
        if self.last_self_assessment and self.last_self_assessment.confidence_level < 0.4:
            return True

        # If the task is complex (estimated by description length)
        if len(task_description) > 500:
            return True

        # If we have a high error rate
        if self.performance_metrics.task_success_rate < 0.6:
            return True

        # If context is getting large
        if context_size > 3000:
            return True

        return False

    def should_change_strategy(self) -> bool:
        """Determine if the agent should change its strategy"""
        # If we've hit the error threshold, consider changing strategy
        if self.consecutive_errors >= self.error_threshold:
            return True
        # If performance metrics are declining, consider changing strategy
        if self.performance_metrics.task_success_rate < 0.5:
            return True
        # If we've been in the same state for too long, consider changing strategy
        if self.get_state_duration(self.current_state) > 600:
            return True
        return False


class MetacognitiveAgent(ContextAwareAgent):
    """Agent with metacognitive capabilities"""

    def __init__(
        self,
        model: Chatterer,
        agent_id: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        delegation_enabled: bool = True,
        parent_agent: Optional[Any] = None,  # Using Any to avoid circular import
        shared_memory: Optional[Any] = None,  # Using Any to avoid circular import
        memory_access: Any = None,  # Using Any to avoid circular import
        max_tokens: int = 8000,
        warning_threshold: float = 0.8,
        metacognition_enabled: bool = True,
        assessment_interval: float = 300.0,
    ):
        # Initialize parent class
        super().__init__(
            model=model,
            agent_id=agent_id,
            tools=tools,
            delegation_enabled=delegation_enabled,
            parent_agent=parent_agent,
            shared_memory=shared_memory,
            memory_access=memory_access,
            max_tokens=max_tokens,
            warning_threshold=warning_threshold,
        )

        # Initialize metacognitive monitor
        self.metacognition_enabled = metacognition_enabled
        self.metacognitive_monitor = MetacognitiveMonitor(model)
        self.metacognitive_monitor.assessment_interval = assessment_interval

        # Register metacognitive tools
        self.register_function_as_tool(
            self.get_performance_metrics,
            name="get_performance_metrics",
            description="Get performance metrics for the agent",
        )

        self.register_function_as_tool(
            self.create_task_plan,
            name="create_task_plan",
            description="Create a task execution plan with steps and estimated resource usage",
        )

        self.register_function_as_tool(
            self.perform_self_assessment,
            name="perform_self_assessment",
            description="Perform a self-assessment of agent performance and capabilities",
        )

        self.register_function_as_tool(
            self.get_strategy_adjustment,
            name="get_strategy_adjustment",
            description="Get recommendations for strategy adjustments based on performance",
        )

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the agent"""
        metrics = self.metacognitive_monitor.performance_metrics
        return {
            "task_success_rate": metrics.task_success_rate,
            "average_task_time": metrics.average_task_time,
            "tasks_completed": metrics.tasks_completed,
            "tasks_failed": metrics.tasks_failed,
            "total_tokens_used": metrics.total_tokens_used,
            "tool_usage_counts": metrics.tool_usage_counts,
            "error_counts": metrics.error_counts,
        }

    async def create_task_plan(self, task_description: str, priority: int = 3) -> Dict[str, Any]:
        """Create a task execution plan with steps and estimated resource usage"""
        task_id = self.metacognitive_monitor.create_task_plan(task_description, priority)
        plan = self.metacognitive_monitor.task_plans[task_id]
        return {
            "task_id": plan.task_id,
            "steps": plan.steps,
            "estimated_tokens": plan.estimated_tokens,
            "estimated_time": plan.estimated_time,
            "priority": plan.priority,
            "status": plan.status,
        }

    async def perform_self_assessment(self) -> Dict[str, Any]:
        """Perform a self-assessment of agent performance and capabilities"""
        # Get recent conversation from context manager
        conversation = self.context_manager.get_conversation_for_model() if hasattr(self, "context_manager") else []

        # Perform self-assessment
        assessment = await self.metacognitive_monitor.perform_self_assessment(conversation)

        return {
            "assessment": {
                "strengths": assessment.strengths,
                "weaknesses": assessment.weaknesses,
                "improvement_areas": assessment.improvement_areas,
                "confidence_level": assessment.confidence_level,
                "reasoning": assessment.reasoning,
            }
        }

    async def get_strategy_adjustment(self) -> Dict[str, Any]:
        """Get recommendations for strategy adjustments based on performance"""
        recommendations = self.metacognitive_monitor.get_recommendations()
        should_change = self.metacognitive_monitor.should_change_strategy()

        return {
            "should_change_strategy": should_change,
            "recommendations": recommendations,
            "current_state": self.metacognitive_monitor.current_state.name,
        }

    async def execute_tool(self, tool_name: str, **parameters: Any) -> Any:
        """Execute a tool and record usage for metacognition"""
        result = await super().execute_tool(tool_name, **parameters)

        # Record tool usage for metacognition
        if self.metacognition_enabled:
            self.metacognitive_monitor.record_tool_usage(tool_name)
            if not result.success:
                self.metacognitive_monitor.record_error(f"Tool error: {tool_name}")
            else:
                self.metacognitive_monitor.record_success()

        return result

    async def run(self, prompt: str) -> str:
        """Run the agent with a prompt, with metacognitive monitoring"""
        if self.metacognition_enabled:
            self.metacognitive_monitor.set_state(AgentState.PLANNING)

        # Run the agent using parent class method
        response = await super().run(prompt)

        if self.metacognition_enabled:
            self.metacognitive_monitor.set_state(AgentState.IDLE)

        return response
