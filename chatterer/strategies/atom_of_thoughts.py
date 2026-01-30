"""
Atom-of-Thought (AoT) reasoning pipeline using LangGraph.

Inspired by https://github.com/qixucen/atom
"""

from __future__ import annotations

import asyncio
import logging
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, NotRequired, Optional, Required, TypedDict, Union

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.graph.state import CompiledStateGraph  # pyright: ignore[reportMissingTypeStubs]
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..language_model import Chatterer

# Type aliases for message handling
MessageDict = dict[str, Any]  # {"role": "user", "content": "..."}
Message = Union[BaseMessage, MessageDict]
MessageList = list[Message]

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str], None]


def extract_question_text(messages: LanguageModelInput) -> str:
    """
    Extract the question text (last user message content) from LanguageModelInput.

    Args:
        messages: String, list of dicts, or list of BaseMessages

    Returns:
        The text content of the last user message (for use in prompt templates)
    """
    from langchain_core.messages import BaseMessage

    def _get_text_content(content: Any) -> str:  # pyright: ignore[reportUnknownParameterType,reportMissingParameterType]
        """Extract text from content (handles str or multimodal list)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Multimodal content - extract text parts
            texts: list[str] = [item.get("text", "") if isinstance(item, dict) else str(item) for item in content]  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType,reportUnknownVariableType]
            return " ".join(texts)
        return str(content)  # pyright: ignore[reportUnknownArgumentType]

    # Simple string case
    if isinstance(messages, str):
        return messages

    if not messages:
        raise ValueError("messages list cannot be empty")

    # Find last user message and extract its text content
    for msg in reversed(list(messages)):
        if isinstance(msg, BaseMessage):
            if msg.type in ("human", "user"):
                return _get_text_content(msg.content)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        elif isinstance(msg, dict):
            if msg.get("role") in ("user", "human"):
                return _get_text_content(msg.get("content", ""))

    raise ValueError("messages must contain at least one user message")


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

    - qa_timeout: Per-LLM-call timeout (with streaming partial capture)
    - decomposition_timeout: Total time for the decomposition branch

    All timeouts use streaming to capture partial responses when timeout occurs.

    Args:
        qa_timeout: Seconds per individual LLM call (None = no limit)
        decomposition_timeout: Seconds for entire decomposition process (None = no limit)
    """

    qa_timeout: float | None = None
    decomposition_timeout: float | None = None

    def is_enabled(self) -> bool:
        """Return True if any timeout is configured."""
        return self.qa_timeout is not None or self.decomposition_timeout is not None

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

    def pretty_print(self) -> None:
        """Print formatted results using rich."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        console = Console()

        # Final Answer Panel
        confidence_color = "green" if self.confidence >= 0.7 else "yellow" if self.confidence >= 0.4 else "red"
        answer_text = Text()
        answer_text.append(self.answer)
        answer_text.append("\n\nConfidence: ", style="dim")
        answer_text.append(f"{self.confidence:.2f}", style=f"bold {confidence_color}")

        console.print(Panel(answer_text, title="[bold cyan]Final Answer[/]", border_style="cyan"))

        # Intermediate Results
        if self.direct_answer or self.decompose_answer or self.sub_questions or self.contracted_question:
            console.print()

            if self.direct_answer:
                console.print(Panel(self.direct_answer, title="[bold blue]Direct Answer[/]", border_style="blue"))

            if self.decompose_answer:
                console.print(
                    Panel(self.decompose_answer, title="[bold magenta]Decomposed Answer[/]", border_style="magenta")
                )

            if self.sub_questions:
                table = Table(
                    title=f"Sub-questions ({len(self.sub_questions)})", show_header=True, header_style="bold green"
                )
                table.add_column("#", style="dim", width=3)
                table.add_column("Question", style="cyan")
                table.add_column("Answer", style="white")

                for i, sq in enumerate(self.sub_questions, 1):
                    table.add_row(str(i), sq.question, sq.answer or "[dim]No answer[/]")

                console.print(table)

            if self.contracted_question:
                console.print(
                    Panel(self.contracted_question, title="[bold yellow]Contracted Question[/]", border_style="yellow")
                )

        # Timeout warning if applicable
        if self.timed_out:
            console.print(
                Panel(f"Reason: {self.timeout_reason.value}", title="[bold red]Timeout Occurred[/]", border_style="red")
            )


def _empty_sub_questions() -> list[SubQuestion]:
    return []


class _DecomposeResult(BaseModel):
    """Result of decomposition - only sub_questions, no premature synthesis."""

    reasoning: str
    sub_questions: list[SubQuestion] = Field(default_factory=_empty_sub_questions)


class _SynthesizeResult(BaseModel):
    """Result of synthesizing answers from answered sub_questions."""

    reasoning: str
    answer: str


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
    messages: LanguageModelInput  # Full conversation context (preserved as-is)
    direct_answer: str
    decompose_answer: str
    sub_questions: list[SubQuestion]
    contracted_question: str
    answer: str
    confidence: float
    timeout_info: AoTTimeoutInfo
    _timeout_config: AoTTimeoutConfig  # Internal, passed through state


class _DecomposeOutput(TypedDict):
    """Output from recursive decomposition."""

    answer: str
    sub_questions: list[SubQuestion]
    timeout_info: AoTTimeoutInfo


class PartialResultCollector:
    """
    Mutable container that survives asyncio.CancelledError.

    Used to collect partial results during recursive decomposition,
    so that when decomposition_timeout fires, we can recover whatever
    work was completed before cancellation.

    All methods include try/except guards for graceful degradation.
    """

    def __init__(self) -> None:
        self.sub_questions: list[SubQuestion] = []
        self.completed_paths: list[str] = []
        self.timed_out_paths: list[str] = []
        self.synthesized_answer: str = ""
        self._lock: asyncio.Lock = asyncio.Lock()

    async def add_sub_question(self, sq: SubQuestion, path: str) -> None:
        """Record a sub-question (with or without answer). Fails silently on error."""
        try:
            async with self._lock:
                # Update existing or add new
                existing = next((s for s in self.sub_questions if s.question == sq.question), None)
                if existing:
                    existing.answer = sq.answer
                else:
                    self.sub_questions.append(sq.model_copy())
        except Exception as e:
            logger.debug(f"PartialResultCollector.add_sub_question failed for path={path}: {e}")

    async def mark_path_completed(self, path: str) -> None:
        """Mark a path as completed. Fails silently on error."""
        try:
            async with self._lock:
                if path not in self.completed_paths:
                    self.completed_paths.append(path)
        except Exception as e:
            logger.debug(f"PartialResultCollector.mark_path_completed failed for path={path}: {e}")

    async def mark_path_timeout(self, path: str) -> None:
        """Mark a path as timed out. Fails silently on error."""
        try:
            async with self._lock:
                if path not in self.timed_out_paths:
                    self.timed_out_paths.append(path)
        except Exception as e:
            logger.debug(f"PartialResultCollector.mark_path_timeout failed for path={path}: {e}")

    async def set_synthesized_answer(self, answer: str) -> None:
        """Record synthesized answer. Fails silently on error."""
        try:
            async with self._lock:
                if answer:  # Only update if non-empty
                    self.synthesized_answer = answer
        except Exception as e:
            logger.debug(f"PartialResultCollector.set_synthesized_answer failed: {e}")

    def to_output(self) -> _DecomposeOutput:
        """Convert collected results to output format."""
        return {
            "answer": self.synthesized_answer,
            "sub_questions": self.sub_questions.copy(),
            "timeout_info": {
                "timed_out": True,
                "reason": TimeoutReason.DECOMPOSITION,
                "completed_paths": self.completed_paths.copy(),
                "timed_out_paths": self.timed_out_paths.copy(),
            },
        }


# ==================== Prompts ====================


_DIRECT_PROMPT = """Answer directly with brief reasoning.
Return JSON: {{"reasoning": "...", "answer": "..."}}

Question: {question}"""

_DECOMPOSE_PROMPT = """Break down the question into sub-questions that would help answer it.
Return JSON: {{"reasoning": "...", "sub_questions": [{{"question": "...", "answer": null}}]}}

Question: {question}"""

_SYNTHESIZE_PROMPT = """Based on the sub-questions and their answers, provide a comprehensive answer to the original question.
Return JSON: {{"reasoning": "...", "answer": "..."}}

Original question: {question}
Sub-answers:
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
    messages: LanguageModelInput,
    *,
    timeout: AoTTimeoutConfig | float | None = None,
) -> AoTState:
    """
    Prepare input state for AoT graph invocation.

    Args:
        messages: Input as string or message list (supports multi-turn, multimodal).
        timeout: Timeout config, or a single number for decomposition_timeout.

    Returns:
        AoTState dict ready for graph.ainvoke() / graph.invoke().

    Example:
        state = aot_input("What causes seasons?", timeout=30)
        result = await graph.ainvoke(state)
    """
    question_text = extract_question_text(messages)

    state: AoTState = {
        "question": question_text,
        "messages": messages,
    }

    if timeout is not None:
        if isinstance(timeout, (int, float)):
            # Single number: use as decomposition_timeout
            timeout = AoTTimeoutConfig(decomposition_timeout=timeout)
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
        instruction: str,
        response_model: type[T],
        context: MessageList | None = None,
        last_message: Message | None = None,
    ) -> T | None:
        """Async generate with system instruction, preserving multimodal content.

        Args:
            instruction: System instruction for the LLM.
            response_model: The Pydantic model to parse the response into.
            context: Optional prior messages (conversation history).
            last_message: The last user message (may contain multimodal content).

        Returns:
            Parsed response model or None on error.
        """
        try:
            # Build: [system instruction] + [context] + [last_message]
            messages: MessageList = [{"role": "system", "content": instruction}]
            if context:
                messages.extend(context)
            if last_message:
                messages.append(last_message)

            result = await chatterer.agenerate_pydantic(
                messages=messages,
                response_model=response_model,
            )
            return result
        except Exception:
            import traceback

            traceback.print_exc()
            return None

    async def _async_safe_generate_with_partial[T: BaseModel](
        instruction: str,
        response_model: type[T],
        timeout: float | None = None,
        context: MessageList | None = None,
        last_message: Message | None = None,
    ) -> tuple[T | None, bool]:
        """Stream-based generation with partial capture, preserving multimodal content.

        Args:
            instruction: System instruction for the LLM.
            response_model: The Pydantic model to parse the response into.
            timeout: Optional timeout in seconds.
            context: Optional prior messages (conversation history).
            last_message: The last user message (may contain multimodal content).

        Returns:
            (result, timed_out): The result (possibly partial) and whether timeout occurred.
        """
        # Build: [system instruction] + [context] + [last_message]
        messages: MessageList = [{"role": "system", "content": instruction}]
        if context:
            messages.extend(context)
        if last_message:
            messages.append(last_message)

        last_valid: T | None = None

        async def collect_stream() -> T | None:
            nonlocal last_valid
            try:
                async for chunk in chatterer.agenerate_pydantic_stream(
                    messages=messages,
                    response_model=response_model,
                ):
                    last_valid = chunk
                return last_valid
            except Exception:
                import traceback

                traceback.print_exc()
                return last_valid  # Return partial even on error

        try:
            if timeout:
                result = await asyncio.wait_for(collect_stream(), timeout=timeout)
                return (result, False)
            else:
                result = await collect_stream()
                return (result, False)
        except asyncio.TimeoutError:
            return (last_valid, True)  # Return partial on timeout

    async def _async_decompose_recursive(
        question: str,
        depth: int,
        path: str = "root",
        timeout_config: AoTTimeoutConfig | None = None,
        semaphore: asyncio.Semaphore | None = None,
        context: MessageList | None = None,
        last_message: Message | None = None,
        collector: PartialResultCollector | None = None,
    ) -> _DecomposeOutput:
        """Async recursive decomposition with native asyncio concurrency control.

        Args:
            question: Question text (for prompt templates and sub-question text)
            context: Conversation history (excluding last message)
            last_message: Original last message (may be multimodal at root level,
                         or simple text dict for sub-questions)
        """
        if depth <= 0:
            _report("subq", f"LEAF|{path}|{question}")
            qa_timeout = timeout_config.qa_timeout if timeout_config else None
            # For leaf nodes, use the instruction + context + last_message
            result, timed_out = await _async_safe_generate_with_partial(
                _DIRECT_PROMPT.format(question=question),
                _DirectAnswer,
                timeout=qa_timeout,
                context=context,
                last_message=last_message,
            )
            answer = result.answer if result else ""
            # Record result in collector IMMEDIATELY (before any subsequent await)
            if collector:
                leaf_sq = SubQuestion(question=question, answer=answer if answer else None)
                await collector.add_sub_question(leaf_sq, path)
                if not timed_out:
                    await collector.mark_path_completed(path)
                else:
                    await collector.mark_path_timeout(path)
            partial_marker = " [PARTIAL]" if timed_out and answer else ""
            if timed_out:
                _report("subq", f"TIMEOUT|{path}|partial={bool(answer)}")
            else:
                _report("subq", f"DONE|{path}|{answer}")
            return {
                "answer": f"{answer}{partial_marker}" if answer else "",
                "sub_questions": [],
                "timeout_info": {
                    "timed_out": timed_out,
                    "reason": TimeoutReason.SUBQUESTION if timed_out else TimeoutReason.NONE,
                },
            }

        _report("decompose", f"DECOMPOSE|depth={depth}|path={path}")

        qa_timeout = timeout_config.qa_timeout if timeout_config else None
        decompose_result, decomp_timed_out = await _async_safe_generate_with_partial(
            _DECOMPOSE_PROMPT.format(question=question),
            _DecomposeResult,
            timeout=qa_timeout,
            context=context,
            last_message=last_message,
        )
        if decomp_timed_out:
            _report("decompose", f"TIMEOUT|path={path}|decomposition|partial={decompose_result is not None}")
            # Use partial sub_questions if available
            partial_subs = decompose_result.sub_questions if decompose_result else []
            return {
                "answer": "",
                "sub_questions": partial_subs,
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
        # Record discovered sub-questions in collector IMMEDIATELY with correct indices
        if collector:
            for idx, sq in enumerate(subs):
                await collector.add_sub_question(sq, f"{path}.{idx}")
        if subs:
            for i, sq in enumerate(subs):
                _report("subq", f"NEW|{path}.{i}|{sq.question}")

        # Track original index alongside unanswered sub-questions
        unanswered_with_idx = [(orig_idx, sq) for orig_idx, sq in enumerate(subs) if not sq.answer]
        if unanswered_with_idx:
            # Create semaphore for concurrency control if not provided
            if not semaphore:
                semaphore = asyncio.Semaphore(max_workers)

            async def process_subquestion(sq: SubQuestion, orig_idx: int) -> dict[str, str]:
                """Process a single subquestion with concurrency control.

                No outer timeout here - relies on inner qa_timeout for partial capture.
                Timeout status propagates through sub_result["timeout_info"].

                Args:
                    sq: The sub-question to process
                    orig_idx: Original index in subs list (for consistent path naming)
                """
                sub_path = f"{path}.{orig_idx}"  # Use original index for path
                async with semaphore:
                    _report("subq", f"START|{sub_path}")
                    try:
                        # For sub-questions, create a text-only message (not multimodal)
                        sub_last_message = {"role": "user", "content": sq.question}
                        sub_result = await _async_decompose_recursive(
                            sq.question, depth - 1, sub_path, timeout_config, semaphore,
                            context=context,
                            last_message=sub_last_message,
                            collector=collector,  # Pass collector for recursive calls
                        )
                        sq.answer = sub_result["answer"]
                        # Update collector IMMEDIATELY after getting answer
                        if collector and sq.answer:
                            await collector.add_sub_question(sq, sub_path)
                        timed_out = sub_result["timeout_info"].get("timed_out", False)
                        status = "timeout" if timed_out else "completed"
                        _report("subq", f"{status.upper()}|{sub_path}")
                        return {"path": sub_path, "status": status}
                    except Exception:
                        sq.answer = ""
                        _report("subq", f"ERROR|{sub_path}")
                        return {"path": sub_path, "status": "error"}

            tasks = [process_subquestion(sq, orig_idx) for orig_idx, sq in unanswered_with_idx]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, dict):
                    if result["status"] == "completed":
                        completed_paths.append(result["path"])
                    elif result["status"] == "timeout":
                        timed_out_paths.append(result["path"])

        # Synthesize answer from answered sub-questions (include partial answers)
        answered_subs = [sq for sq in subs if sq.answer and not sq.answer.startswith("[TIMEOUT")]
        synthesized = ""
        synth_timed_out = False
        if answered_subs:
            sub_answers_str = "\n".join(f"- {sq.question}: {sq.answer}" for sq in answered_subs)
            _report("decompose", f"SYNTHESIZING|path={path}|{len(answered_subs)} answers")
            synth_result, synth_timed_out = await _async_safe_generate_with_partial(
                _SYNTHESIZE_PROMPT.format(question=question, sub_answers=sub_answers_str),
                _SynthesizeResult,
                timeout=qa_timeout,
                context=context,
                last_message=last_message,
            )
            synthesized = synth_result.answer if synth_result else ""
            # Record synthesized answer in collector IMMEDIATELY
            if collector and synthesized:
                await collector.set_synthesized_answer(synthesized)
            if synth_timed_out and synthesized:
                synthesized += " [PARTIAL]"

        _report("decompose", f"DONE|path={path}")
        any_timeout = len(timed_out_paths) > 0 or synth_timed_out
        timeout_info: AoTTimeoutInfo = {
            "timed_out": any_timeout,
            "reason": TimeoutReason.SUBQUESTION if any_timeout else TimeoutReason.NONE,
            "completed_paths": completed_paths,
            "timed_out_paths": timed_out_paths,
        }
        return {"answer": synthesized, "sub_questions": subs, "timeout_info": timeout_info}

    # Node functions (async for native asyncio support)
    async def prepare(state: AoTState) -> AoTState:
        """Initialize the state with the question."""
        question = state.get("question", "")
        messages = state.get("messages", [{"role": "user", "content": question}])

        _report("prepare", f"Processing: {question}")
        result: AoTState = {
            "question": question,
            "messages": messages,
        }
        timeout_cfg = state.get("_timeout_config")
        if timeout_cfg is not None:
            result["_timeout_config"] = timeout_cfg
        return result

    def _split_messages(messages: LanguageModelInput) -> tuple[MessageList | None, Message | None]:
        """Split messages into (context, last_message).

        Returns (context_messages, last_message) where:
        - context_messages: All messages except the last (or None if single message)
        - last_message: The last message (preserves multimodal content)
        """
        if isinstance(messages, str):
            return None, {"role": "user", "content": messages}
        try:
            msg_list: MessageList = list(messages)  # pyright: ignore[reportUnknownArgumentType,reportAssignmentType]
            if not msg_list:
                return None, None
            if len(msg_list) == 1:
                return None, msg_list[0]
            return msg_list[:-1], msg_list[-1]
        except (TypeError, AttributeError):
            return None, None  # PromptValue or other non-sequence type

    async def parallel_paths(state: AoTState) -> AoTState:
        """Run direct answer and decomposition in parallel using asyncio.gather."""
        question = state.get("question", "")
        messages = state.get("messages", "")
        timeout_cfg = state.get("_timeout_config")

        # Split messages into context (history) and last_message (preserves multimodal)
        context, last_message = _split_messages(messages) if messages else (None, None)

        _report("parallel", "START|direct+decompose")

        # Get timeout values
        qa_timeout = timeout_cfg.qa_timeout if timeout_cfg else None
        decomp_timeout = timeout_cfg.decomposition_timeout if timeout_cfg else None

        # Direct answer with qa_timeout (streaming for partial capture)
        async def get_direct() -> _DirectAnswer | None:
            result, _ = await _async_safe_generate_with_partial(
                _DIRECT_PROMPT.format(question=question),
                _DirectAnswer,
                timeout=qa_timeout,
                context=context,
                last_message=last_message,
            )
            return result

        # Decomposition branch with decomposition_timeout
        async def get_decompose() -> _DecomposeOutput:
            # Create collector to capture partial results on timeout
            collector = PartialResultCollector() if decomp_timeout else None

            if decomp_timeout:
                try:
                    return await asyncio.wait_for(
                        _async_decompose_recursive(
                            question, max_depth, "root", timeout_cfg, None,
                            context=context, last_message=last_message,
                            collector=collector,
                        ),
                        timeout=decomp_timeout,
                    )
                except asyncio.TimeoutError:
                    _report("decompose", f"TIMEOUT|decomposition_timeout={decomp_timeout}|partial_subs={len(collector.sub_questions) if collector else 0}")
                    # Return partial results instead of empty
                    if collector:
                        return collector.to_output()
                    return {
                        "answer": "",
                        "sub_questions": [],
                        "timeout_info": {"timed_out": True, "reason": TimeoutReason.DECOMPOSITION},
                    }
            else:
                return await _async_decompose_recursive(
                    question, max_depth, "root", timeout_cfg, None,
                    context=context, last_message=last_message,
                    collector=None,  # No collector needed without timeout
                )

        # Run both paths concurrently
        direct_result, decompose_result = await asyncio.gather(get_direct(), get_decompose())

        direct_answer = direct_result.answer if direct_result else ""
        decompose_answer = decompose_result["answer"]
        sub_questions = decompose_result.get("sub_questions", [])

        _report("parallel", "DONE|direct+decompose")

        # Handle timeout info propagation
        decompose_timeout_info = decompose_result["timeout_info"]
        timed_out = decompose_timeout_info["timed_out"]
        timeout_info: AoTTimeoutInfo = {"timed_out": False, "reason": TimeoutReason.NONE}
        if timed_out:
            timeout_info = {
                "timed_out": True,
                "reason": decompose_timeout_info.get("reason", TimeoutReason.NONE),
                "completed_paths": decompose_timeout_info.get("completed_paths", []),
                "timed_out_paths": decompose_timeout_info.get("timed_out_paths", []),
            }
            timeout_info["partial_results"] = {
                sq.question: sq.answer for sq in sub_questions if sq.answer and not sq.answer.startswith("[TIMEOUT")
            }
            # Add count of recovered items for logging
            timeout_info["active_futures_at_timeout"] = len([
                sq for sq in sub_questions if not sq.answer
            ])

        return {
            "direct_answer": direct_answer,
            "decompose_answer": decompose_answer,
            "sub_questions": sub_questions,
            "timeout_info": timeout_info,
        }

    async def contract(state: AoTState) -> AoTState:
        """Contract phase - async version."""
        question = state.get("question", "")
        messages = state.get("messages", "")
        sub_questions = state.get("sub_questions", [])

        # Split messages for multi-turn scenarios (preserves multimodal in last_message)
        context, last_message = _split_messages(messages) if messages else (None, None)

        _report("contract", f"START|{len(sub_questions)} subs")

        sub_str = "\n".join(
            f"- {sq.question}: {sq.answer}"
            for sq in sub_questions
            if sq.answer and not sq.answer.startswith("[TIMEOUT")
        )

        result = await _async_safe_generate(
            _CONTRACT_PROMPT.format(question=question, sub_answers=sub_str),
            _ContractedQuestion,
            context=context,
            last_message=last_message,
        )

        contracted = result.question if result else question
        _report("contract", f"DONE|{contracted}")

        return {"contracted_question": contracted}

    async def ensemble(state: AoTState) -> AoTState:
        """Ensemble phase - async version."""
        messages = state.get("messages", "")
        direct_answer = state.get("direct_answer", "")
        decompose_answer = state.get("decompose_answer", "")

        # Split messages for multi-turn scenarios (preserves multimodal in last_message)
        context, last_message = _split_messages(messages) if messages else (None, None)

        _report("ensemble", "START")

        prompt = _ENSEMBLE_PROMPT.format(
            question=state.get("contracted_question", state.get("question", "")),
            direct=direct_answer,
            decompose=decompose_answer,
            contracted_question=state.get("contracted_question", ""),
        )

        result = await _async_safe_generate(prompt, _EnsembleResult, context=context, last_message=last_message)

        answer = result.answer if result else direct_answer or decompose_answer
        confidence = result.confidence if result else 0.5

        # Note: timeout_info is preserved in state for caller to inspect
        # We don't artificially reduce confidence - let the caller decide based on timeout_info

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
