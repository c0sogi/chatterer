"""
Atom-of-Thought (AoT) reasoning pipeline using LangGraph.

Inspired by https://github.com/qixucen/atom
"""

from __future__ import annotations

import asyncio
import logging
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, Literal, NotRequired, Optional, Required, TypedDict

from langchain_core.language_models.base import LanguageModelInput
from langgraph.graph import END, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.graph.state import CompiledStateGraph  # pyright: ignore[reportMissingTypeStubs]
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..language_model import Chatterer

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str], None]


def normalize_messages(question: LanguageModelInput) -> tuple[str, list[dict[str, str]]]:
    """
    Normalize question input to (question_text, messages).

    Args:
        question: Either a string, sequence of dicts, or sequence of BaseMessages

    Returns:
        (question_text, full_messages): The question for state + full conversation

    Raises:
        ValueError: If messages list is empty or has no user message
    """
    from collections.abc import Sequence

    from langchain_core.messages import BaseMessage

    # Case 1: Simple string
    if isinstance(question, str):
        return question, [{"role": "user", "content": question}]

    # Case 2: Sequence type (could be list of dicts or BaseMessages)
    if isinstance(question, Sequence):
        if not question:
            raise ValueError("messages list cannot be empty")

        # Check if it's a sequence of BaseMessage objects
        first_item = question[0]
        if isinstance(first_item, BaseMessage):
            # Convert BaseMessage to dict format
            messages: list[dict[str, str]] = []
            for msg in question:
                if isinstance(msg, BaseMessage):
                    # BaseMessage.content can be str or list, normalize to str
                    msg_content = msg.content  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
                    content_str: str
                    if isinstance(msg_content, str):
                        content_str = msg_content
                    else:
                        # Handle list content by converting to string
                        # pyright doesn't know the exact type, so we cast
                        content_str = str(msg_content)  # pyright: ignore[reportUnknownArgumentType]

                    # Map LangChain message types to standard roles
                    role_map = {"human": "user", "ai": "assistant", "system": "system"}
                    role = role_map.get(msg.type, msg.type) if hasattr(msg, "type") else "user"
                    messages.append({"role": role, "content": content_str})
            # Find last user message (check both "user" and "human" for compatibility)
            user_msgs: list[dict[str, str]] = [m for m in messages if m["role"] in ("user", "human")]
            if not user_msgs:
                raise ValueError("messages must contain at least one user message")
            last_content: str = user_msgs[-1]["content"]
            return last_content, messages

        # Case 3: Sequence of dicts
        # Type narrowing: we know question is a sequence of dicts now
        user_messages = [m for m in question if isinstance(m, dict) and m.get("role") == "user"]
        if not user_messages:
            raise ValueError("messages must contain at least one user message")

        last_user = user_messages[-1]
        content = last_user.get("content", "")
        if not content or not isinstance(content, str):
            raise ValueError("user message content must be a non-empty string")

        # Convert all messages to dict format for consistent return type
        result_messages: list[dict[str, str]] = []
        for m in question:
            if isinstance(m, dict):
                result_messages.append(dict(m))  # pyright: ignore[reportUnknownArgumentType]
            else:
                result_messages.append({"role": "user", "content": str(m)})

        return content, result_messages

    # Unsupported type
    raise TypeError(f"Unsupported message input type: {type(question)}")


# ==================== Timeout Types ====================


class TimeoutReason(StrEnum):
    """Categorizes why a timeout occurred."""

    SUBQUESTION = "SUBQUESTION"
    DECOMPOSITION = "DECOMPOSITION"
    NONE = "NONE"


class AoTTimeoutInfo(TypedDict):
    """Metadata about timeout events during AoT execution."""

    timed_out: Required[bool]
    reason: Required[TimeoutReason]  # TimeoutReason value
    elapsed_seconds: NotRequired[float]
    completed_paths: NotRequired[list[str]]
    timed_out_paths: NotRequired[list[str]]
    partial_results: NotRequired[dict[str, str]]
    active_futures_at_timeout: NotRequired[int]  # Count of zombie threads still running


class AoTTimeoutConfig(BaseModel):
    """
    Configuration for AoT timeout behavior.

    Uses native asyncio patterns for all timeout handling:
    - asyncio.wait_for() for individual operation timeouts
    - asyncio.Semaphore for concurrency control
    - No ThreadPoolExecutor, eliminating zombie thread issues

    Note: While timeout handling is clean, the underlying HTTP connection
    to the LLM provider may continue until the provider responds. This is
    a limitation of HTTP, not the timeout implementation.

    Args:
        subquestion_timeout: Seconds per subquestion processing (None = no limit)
        decomposition_timeout: Seconds per decomposition LLM call (None = no limit)
        on_timeout: "abort" stops immediately; "continue_partial" preserves completed work
    """

    subquestion_timeout: float | None = None
    decomposition_timeout: float | None = None
    on_timeout: Literal["abort", "continue_partial"] = "continue_partial"

    def is_enabled(self) -> bool:
        """Return True if any timeout is configured."""
        return any(
            [
                self.subquestion_timeout is not None,
                self.decomposition_timeout is not None,
            ]
        )

    @staticmethod
    def no_timeout() -> "AoTTimeoutConfig":
        """Factory method for no-timeout configuration."""
        return AoTTimeoutConfig()


# ==================== Models ====================


class SubQuestion(BaseModel):
    question: str
    answer: Optional[str] = None


class AoTResult(BaseModel):
    """Clean output from AoT graph execution."""

    question: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)

    # Optional detailed results
    direct_answer: str | None = None
    decompose_answer: str | None = None
    contracted_question: str | None = None
    sub_questions: list[SubQuestion] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]

    # Timeout information
    timed_out: bool = False
    timeout_reason: TimeoutReason = TimeoutReason.NONE
    timeout_info: AoTTimeoutInfo | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for backward compatibility."""
        return self.model_dump()


def _empty_sub_questions() -> list[SubQuestion]:
    return []


class _DecomposeResult(BaseModel):
    reasoning: str
    sub_questions: list[SubQuestion] = Field(default_factory=_empty_sub_questions)
    synthesized_answer: str


class _DirectAnswer(BaseModel):
    reasoning: str
    answer: str


class _ContractedQuestion(BaseModel):
    reasoning: str
    question: str


class _EnsembleResult(BaseModel):
    reasoning: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)


# ==================== State ====================


class AoTState(TypedDict, total=False):
    """State for AoT graph. Also serves as the result type."""

    question: str  # Extracted question text for prompts
    messages: list[dict[str, str]]  # Full conversation context
    direct_answer: str
    decompose_answer: str
    sub_questions: list[SubQuestion]
    contracted_question: str
    answer: str
    confidence: float
    timeout_info: AoTTimeoutInfo
    _timeout_config: AoTTimeoutConfig  # Internal, passed through state
    _start_time: float  # Internal, for global timeout tracking


class _DecomposeOutput(TypedDict):
    """Output from recursive decomposition."""

    answer: str
    sub_questions: list[SubQuestion]
    timeout_info: AoTTimeoutInfo


# ==================== Prompts ====================


_DIRECT_PROMPT = """Answer directly with brief reasoning.
Return JSON: {{"reasoning": "...", "answer": "..."}}

Question: {question}"""

_DECOMPOSE_PROMPT = """Break down into sub-questions and synthesize an answer.
Return JSON: {{"reasoning": "...", "sub_questions": [{{"question": "...", "answer": null}}], "synthesized_answer": "..."}}

Question: {question}
{sub_answers}"""

_CONTRACT_PROMPT = """Simplify into one self-contained question using the sub-answers.
Return JSON: {{"reasoning": "...", "question": "..."}}

Original: {question}
Sub-answers: {sub_answers}"""

_ENSEMBLE_PROMPT = """Select the best answer or synthesize from these approaches.
1. Direct answer: {direct}
2. Decomposed answer: {decompose}
3. Contracted question for context: {contracted_question}

Return JSON: {{"reasoning": "...", "answer": "...", "confidence": 0.0-1.0}}

Original question: {question}"""


# ==================== Input/Output Helpers ====================


def aot_input(
    question: LanguageModelInput,
    *,
    timeout: AoTTimeoutConfig | float | None = None,
) -> AoTState:
    """
    Prepare input state for AoT graph invocation.

    Args:
        question: Question as string or message list (multi-turn conversation).
        timeout: Optional timeout configuration.

    Returns:
        AoTState dict ready for graph.ainvoke() / graph.invoke().

    Example:
        state = aot_input("What causes seasons?", timeout=AoTTimeoutConfig())
        result = await graph.ainvoke(state)
    """
    import time

    question_text, messages = normalize_messages(question)

    state: AoTState = {
        "question": question_text,
        "messages": messages,
        "_start_time": time.time(),
    }

    if timeout is not None:
        if isinstance(timeout, (int, float)):
            timeout = AoTTimeoutConfig(
                subquestion_timeout=timeout,
                decomposition_timeout=timeout,
            )
        state["_timeout_config"] = timeout

    return state


def aot_output(state: AoTState) -> AoTResult:
    """
    Extract clean result from raw graph output state.

    Args:
        state: Raw AoTState from graph.ainvoke() / graph.invoke().

    Returns:
        AoTResult with typed fields and convenience methods.

    Example:
        result = await graph.ainvoke(state)
        output = aot_output(result)
        print(output.answer)
        print(output.confidence)
    """
    timeout_info_raw = state.get("timeout_info")
    timed_out = False
    timeout_reason = TimeoutReason.NONE
    timeout_info: AoTTimeoutInfo | None = None

    if timeout_info_raw:
        timed_out = timeout_info_raw.get("timed_out", False)
        timeout_reason = timeout_info_raw.get("reason", TimeoutReason.NONE)
        if timed_out:
            # Cast to proper type since we know the shape
            timeout_info = timeout_info_raw  # pyright: ignore[reportAssignmentType]

    return AoTResult(
        question=state.get("question", ""),
        answer=state.get("answer", ""),
        confidence=state.get("confidence", 0.5),
        direct_answer=state.get("direct_answer"),
        decompose_answer=state.get("decompose_answer"),
        contracted_question=state.get("contracted_question"),
        sub_questions=state.get("sub_questions", []),
        timed_out=timed_out,
        timeout_reason=timeout_reason,
        timeout_info=timeout_info,
    )


# ==================== Graph Builder ====================


def aot_graph(
    chatterer: Chatterer,
    *,
    max_depth: int = 2,
    max_sub_questions: int = 3,
    max_workers: int = 4,
    on_progress: ProgressCallback | None = None,
) -> CompiledStateGraph[AoTState]:  # pyright: ignore[reportMissingTypeArgument]
    """
    Create an Atom-of-Thought reasoning graph.

    Args:
        chatterer: The Chatterer instance for LLM calls.
        max_depth: Maximum recursion depth for decomposition.
        max_sub_questions: Maximum sub-questions per level.
        max_workers: Maximum parallel workers.
        on_progress: Optional progress callback.

    Returns:
        Compiled LangGraph. Use with aot_input/aot_run/aot_output helpers.

    Example:
        # Basic usage (sync)
        graph = aot_graph(chatterer)
        state = aot_input("What causes seasons?")
        raw = aot_run_sync(graph, state)
        result = aot_output(raw)
        print(result.answer)

        # With timeout controls
        timeout_config = AoTTimeoutConfig(
            subquestion_timeout=10.0,
            decomposition_timeout=5.0,
        )
        state = aot_input("Complex question?", timeout=timeout_config)
        raw = aot_run_sync(graph, state)
        result = aot_output(raw)
        if result.timed_out:
            print(f"Timed out: {result.timeout_reason}")

        # Async usage
        graph = aot_graph(chatterer)
        state = aot_input("What causes seasons?")
        raw = await aot_run(graph, state)
        result = aot_output(raw)
        print(result.answer)
        print(result.confidence)
    """

    def _report(stage: str, message: str) -> None:
        if on_progress:
            on_progress(stage, message)

    async def _async_safe_generate[T: BaseModel](
        prompt: str,
        response_model: type[T],
        context_messages: list[dict[str, str]] | None = None,
    ) -> T | None:
        """Async version of generate_pydantic with exception handling.

        Args:
            prompt: The prompt to send as the final user message.
            response_model: The Pydantic model to parse the response into.
            context_messages: Optional list of prior messages to prepend for context.
                             The prompt is appended as a final user message.

        Returns:
            Parsed response model or None on error.
        """
        try:
            # Build messages: context (if any) + prompt as final user message
            messages: list[dict[str, str]] = list(context_messages) if context_messages else []
            messages.append({"role": "user", "content": prompt})

            result = await chatterer.agenerate_pydantic(
                messages=messages,
                response_model=response_model,
            )
            return result
        except Exception as _e:
            import traceback

            traceback.print_exc()
            return None

    async def _async_decompose_recursive(
        question: str,
        depth: int,
        path: str = "root",
        timeout_config: AoTTimeoutConfig | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> _DecomposeOutput:
        """Async recursive decomposition with native asyncio concurrency control."""
        if depth <= 0:
            _report("subq", f"LEAF|{path}|{question}")
            result = await _async_safe_generate(_DIRECT_PROMPT.format(question=question), _DirectAnswer)
            answer = result.answer if result else ""
            _report("subq", f"DONE|{path}|{answer}")
            return {
                "answer": answer,
                "sub_questions": [],
                "timeout_info": {"timed_out": False, "reason": TimeoutReason.NONE},
            }

        _report("decompose", f"DECOMPOSE|depth={depth}|path={path}")

        # Decompose with optional timeout
        decomp_timeout = timeout_config.decomposition_timeout if timeout_config else None
        try:
            if decomp_timeout:
                decompose_result = await asyncio.wait_for(
                    _async_safe_generate(_DECOMPOSE_PROMPT.format(question=question, sub_answers=""), _DecomposeResult),
                    timeout=decomp_timeout,
                )
            else:
                decompose_result = await _async_safe_generate(
                    _DECOMPOSE_PROMPT.format(question=question, sub_answers=""), _DecomposeResult
                )
        except asyncio.TimeoutError:
            _report("decompose", f"TIMEOUT|path={path}|decomposition")
            return {
                "answer": "",
                "sub_questions": [],
                "timeout_info": {"timed_out": True, "reason": TimeoutReason.DECOMPOSITION},
            }

        if not decompose_result:
            _report("subq", f"DONE|{path}")
            return {
                "answer": "",
                "sub_questions": [],
                "timeout_info": {"timed_out": False, "reason": TimeoutReason.NONE},
            }

        completed_paths: list[str] = []
        timed_out_paths: list[str] = []

        subs = decompose_result.sub_questions[:max_sub_questions]
        if subs:
            for i, sq in enumerate(subs):
                _report("subq", f"NEW|{path}.{i}|{sq.question}")

        unanswered = [sq for sq in subs if not sq.answer]
        if unanswered:
            # Create semaphore for concurrency control if not provided
            if not semaphore:
                semaphore = asyncio.Semaphore(max_workers)

            async def process_subquestion(sq: SubQuestion, sub_path: str) -> dict[str, str]:
                """Process a single subquestion with timeout and concurrency control."""
                async with semaphore:
                    _report("subq", f"START|{sub_path}")
                    try:
                        subq_timeout = timeout_config.subquestion_timeout if timeout_config else None
                        if subq_timeout:
                            sub_result = await asyncio.wait_for(
                                _async_decompose_recursive(sq.question, depth - 1, sub_path, timeout_config, semaphore),
                                timeout=subq_timeout,
                            )
                        else:
                            sub_result = await _async_decompose_recursive(
                                sq.question, depth - 1, sub_path, timeout_config, semaphore
                            )

                        sq.answer = sub_result["answer"]
                        _report("subq", f"DONE|{sub_path}")
                        return {"path": sub_path, "status": "completed"}
                    except asyncio.TimeoutError:
                        sq.answer = "[TIMEOUT]"
                        _report("subq", f"TIMEOUT|{sub_path}|subquestion")
                        return {"path": sub_path, "status": "timeout"}
                    except Exception:
                        sq.answer = ""
                        _report("subq", f"ERROR|{sub_path}")
                        return {"path": sub_path, "status": "error"}

            tasks = [process_subquestion(sq, f"{path}.{i}") for i, sq in enumerate(unanswered)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, dict):
                    if result["status"] == "completed":
                        completed_paths.append(result["path"])
                    elif result["status"] == "timeout":
                        timed_out_paths.append(result["path"])

        _report("decompose", f"SYNTHESIZED|path={path}")
        synthesized = getattr(decompose_result, "synthesized_answer", "")
        timeout_info: AoTTimeoutInfo = {
            "timed_out": len(timed_out_paths) > 0,
            "reason": TimeoutReason.SUBQUESTION if timed_out_paths else TimeoutReason.NONE,
            "completed_paths": completed_paths,
            "timed_out_paths": timed_out_paths,
            "active_futures_at_timeout": 0,  # No zombie threads in async!
        }
        return {"answer": synthesized, "sub_questions": subs, "timeout_info": timeout_info}

    # Node functions (async for native asyncio support)
    async def prepare(state: AoTState) -> AoTState:
        """Initialize the state with the question and timeout tracking."""
        import time

        question = state.get("question", "")
        messages = state.get("messages", [{"role": "user", "content": question}])

        _report("prepare", f"Processing: {question}")
        result: AoTState = {
            "question": question,
            "messages": messages,
            "_start_time": time.time(),
        }
        timeout_cfg = state.get("_timeout_config")
        if timeout_cfg is not None:
            result["_timeout_config"] = timeout_cfg
        return result

    async def parallel_paths(state: AoTState) -> AoTState:
        """Run direct answer and decomposition in parallel using asyncio.gather."""
        question = state.get("question", "")
        messages = state.get("messages", [])
        timeout_cfg = state.get("_timeout_config")

        # Extract context (all messages except the last user message with the question)
        # This provides conversation history for multi-turn scenarios
        context_msgs = messages[:-1] if len(messages) > 1 else None

        _report("parallel", "START|direct+decompose")

        # Run both paths concurrently with native asyncio
        direct_task = _async_safe_generate(_DIRECT_PROMPT.format(question=question), _DirectAnswer, context_msgs)
        decompose_task = _async_decompose_recursive(question, max_depth, "root", timeout_cfg)

        direct_result, decompose_result = await asyncio.gather(direct_task, decompose_task)

        direct_answer = direct_result.answer if direct_result else ""
        decompose_answer = decompose_result["answer"]
        sub_questions = decompose_result.get("sub_questions", [])

        _report("parallel", "DONE|direct+decompose")

        # Handle timeout info propagation
        timeout_info: AoTTimeoutInfo = {"timed_out": False, "reason": TimeoutReason.NONE}
        if decompose_result.get("timeout_info", {}).get("timed_out"):
            timeout_info = {
                "timed_out": True,
                "reason": decompose_result["timeout_info"].get("reason", TimeoutReason.NONE),
                "completed_paths": decompose_result["timeout_info"].get("completed_paths", []),
                "timed_out_paths": decompose_result["timeout_info"].get("timed_out_paths", []),
                "active_futures_at_timeout": 0,
            }
            timeout_info["partial_results"] = {
                sq.question: sq.answer for sq in sub_questions if sq.answer and sq.answer != "[TIMEOUT]"
            }

        return {
            "direct_answer": direct_answer,
            "decompose_answer": decompose_answer,
            "sub_questions": sub_questions,
            "timeout_info": timeout_info,
        }

    async def contract(state: AoTState) -> AoTState:
        """Contract phase - async version."""
        question = state.get("question", "")
        messages = state.get("messages", [])
        sub_questions = state.get("sub_questions", [])

        # Extract context for multi-turn scenarios
        context_msgs = messages[:-1] if len(messages) > 1 else None

        _report("contract", f"START|{len(sub_questions)} subs")

        sub_str = "\n".join(
            f"- {sq.question}: {sq.answer}" for sq in sub_questions if sq.answer and sq.answer != "[TIMEOUT]"
        )

        result = await _async_safe_generate(
            _CONTRACT_PROMPT.format(question=question, sub_answers=sub_str),
            _ContractedQuestion,
            context_msgs,
        )

        contracted = result.question if result else question
        _report("contract", f"DONE|{contracted}")

        return {"contracted_question": contracted}

    async def ensemble(state: AoTState) -> AoTState:
        """Ensemble phase - async version."""
        messages = state.get("messages", [])
        direct_answer = state.get("direct_answer", "")
        decompose_answer = state.get("decompose_answer", "")

        # Extract context for multi-turn scenarios
        context_msgs = messages[:-1] if len(messages) > 1 else None

        _report("ensemble", "START")

        prompt = _ENSEMBLE_PROMPT.format(
            question=state.get("contracted_question", state.get("question", "")),
            direct=direct_answer,
            decompose=decompose_answer,
            contracted_question=state.get("contracted_question", ""),
        )

        result = await _async_safe_generate(prompt, _EnsembleResult, context_msgs)

        answer = result.answer if result else direct_answer or decompose_answer
        confidence = result.confidence if result else 0.5

        # Reduce confidence if timeout occurred
        if state.get("timeout_info", {}).get("timed_out"):
            confidence = confidence * 0.7

        _report("ensemble", f"DONE|confidence={confidence:.2f}")

        return {"answer": answer, "confidence": confidence}

    # Build graph
    graph: StateGraph[AoTState] = StateGraph(AoTState)  # pyright: ignore[reportMissingTypeArgument]
    graph.add_node("prepare", prepare)  # pyright: ignore[reportUnknownMemberType]
    graph.add_node("parallel_paths", parallel_paths)  # pyright: ignore[reportUnknownMemberType]
    graph.add_node("contract", contract)  # pyright: ignore[reportUnknownMemberType]
    graph.add_node("ensemble", ensemble)  # pyright: ignore[reportUnknownMemberType]

    graph.set_entry_point("prepare")  # pyright: ignore[reportUnknownMemberType]
    graph.add_edge("prepare", "parallel_paths")  # pyright: ignore[reportUnknownMemberType]
    graph.add_edge("parallel_paths", "contract")  # pyright: ignore[reportUnknownMemberType]
    graph.add_edge("contract", "ensemble")  # pyright: ignore[reportUnknownMemberType]
    graph.add_edge("ensemble", END)  # pyright: ignore[reportUnknownMemberType]

    return graph.compile()  # pyright: ignore[reportUnknownMemberType]
