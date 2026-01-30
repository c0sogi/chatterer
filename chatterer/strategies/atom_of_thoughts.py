"""
Atom-of-Thought (AoT) reasoning pipeline using LangGraph.

Inspired by https://github.com/qixucen/atom
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Callable, Optional, TypedDict

from langgraph.graph import END, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.graph.state import CompiledStateGraph  # pyright: ignore[reportMissingTypeStubs]
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..language_model import Chatterer

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str], None]


# ==================== Models ====================


class SubQuestion(BaseModel):
    question: str
    answer: Optional[str] = None


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

    question: str
    direct_answer: str
    decompose_answer: str
    sub_questions: list[SubQuestion]
    contracted_question: str
    answer: str
    confidence: float


class _DecomposeOutput(TypedDict):
    """Output from recursive decomposition."""

    answer: str
    sub_questions: list[SubQuestion]


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
        Compiled LangGraph. Use .invoke({"question": "..."}) to run.

    Example:
        graph = aot_graph(chatterer)
        result = graph.invoke({"question": "What causes seasons?"})
        print(result["answer"])
        print(result["confidence"])
        print(result["sub_questions"])
    """

    def _report(stage: str, message: str) -> None:
        if on_progress:
            on_progress(stage, message)

    def _safe_generate[T: BaseModel](prompt: str, response_model: type[T]) -> T | None:
        try:
            return chatterer.generate_pydantic(
                messages=[{"role": "user", "content": prompt}],
                response_model=response_model,
            )
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return None

    def _decompose_recursive(question: str, depth: int, path: str = "root") -> _DecomposeOutput:
        if depth <= 0:
            _report("subq", f"LEAF|{path}|{question}")
            result = _safe_generate(_DIRECT_PROMPT.format(question=question), _DirectAnswer)
            answer = result.answer if result else ""
            _report("subq", f"DONE|{path}|{answer}")
            return {"answer": answer, "sub_questions": []}

        _report("decompose", f"DECOMPOSE|depth={depth}|path={path}")
        result = _safe_generate(_DECOMPOSE_PROMPT.format(question=question, sub_answers=""), _DecomposeResult)
        if not result:
            _report("subq", f"DONE|{path}")
            return {"answer": "", "sub_questions": []}

        subs = result.sub_questions[:max_sub_questions]
        if subs:
            # Report sub-questions with their text (truncated)
            for i, sq in enumerate(subs):
                _report("subq", f"NEW|{path}.{i}|{sq.question}")

        unanswered = [sq for sq in subs if not sq.answer]
        if unanswered:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures: dict[Any, tuple[SubQuestion, str]] = {}
                for i, sq in enumerate(unanswered):
                    sub_path = f"{path}.{i}"
                    _report("subq", f"START|{sub_path}")
                    futures[executor.submit(_decompose_recursive, sq.question, depth - 1, sub_path)] = (sq, sub_path)

                for future in as_completed(futures):
                    sq, sub_path = futures[future]
                    try:
                        sq.answer = future.result()["answer"]
                        _report("subq", f"DONE|{sub_path}")
                    except Exception:
                        sq.answer = ""
                        _report("subq", f"ERROR|{sub_path}")

        _report("decompose", f"SYNTHESIZED|path={path}")
        return {"answer": result.synthesized_answer, "sub_questions": subs}

    # Node functions
    def prepare(state: AoTState) -> AoTState:
        question = state.get("question", "")
        _report("prepare", f"Processing: {question}")
        return {"question": question}

    def parallel_paths(state: AoTState) -> AoTState:
        question = state.get("question", "")
        _report("parallel", "Running direct and decompose...")

        with ThreadPoolExecutor(max_workers=2) as executor:
            direct_future = executor.submit(
                lambda: _safe_generate(_DIRECT_PROMPT.format(question=question), _DirectAnswer)
            )
            decompose_future = executor.submit(lambda: _decompose_recursive(question, max_depth, "root"))

            direct_result = direct_future.result()
            decompose_result = decompose_future.result()

        direct_answer = direct_result.answer if direct_result else ""
        _report("direct", f"Direct: {direct_answer}")
        _report("direct", "COMPLETE")
        _report("decompose", "COMPLETE")

        return {
            "direct_answer": direct_answer,
            "decompose_answer": decompose_result["answer"],
            "sub_questions": decompose_result["sub_questions"],
        }

    def contract(state: AoTState) -> AoTState:
        question = state.get("question", "")
        sub_questions = state.get("sub_questions", [])

        if not sub_questions:
            return {"contracted_question": question}

        _report("contract", f"Contracting {len(sub_questions)} sub-answers...")
        sub_str = "\n".join(f"- {sq.question}: {sq.answer}" for sq in sub_questions if sq.answer)
        result = _safe_generate(_CONTRACT_PROMPT.format(question=question, sub_answers=sub_str), _ContractedQuestion)
        return {"contracted_question": result.question if result else question}

    def ensemble(state: AoTState) -> AoTState:
        _report("ensemble", "Selecting best answer...")
        prompt = _ENSEMBLE_PROMPT.format(
            question=state.get("question", ""),
            direct=state.get("direct_answer", ""),
            decompose=state.get("decompose_answer", ""),
            contracted_question=state.get("contracted_question", ""),
        )
        result = _safe_generate(prompt, _EnsembleResult)

        if result:
            answer, confidence = result.answer, result.confidence
        else:
            answer = state.get("decompose_answer", "") or state.get("direct_answer", "")
            confidence = 0.5

        _report("ensemble", f"Done (confidence: {confidence:.2f})")
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


# ==================== Convenience ====================


def aot_invoke(
    chatterer: Chatterer,
    question: str,
    *,
    max_depth: int = 2,
    max_sub_questions: int = 3,
    max_workers: int = 4,
    on_progress: ProgressCallback | None = None,
) -> AoTState:
    """Convenience function: create graph and invoke."""
    graph = aot_graph(
        chatterer,
        max_depth=max_depth,
        max_sub_questions=max_sub_questions,
        max_workers=max_workers,
        on_progress=on_progress,
    )
    return graph.invoke({"question": question})  # pyright: ignore[reportUnknownMemberType, reportReturnType]


async def aot_ainvoke(
    chatterer: Chatterer,
    question: str,
    *,
    max_depth: int = 2,
    max_sub_questions: int = 3,
    max_workers: int = 4,
    on_progress: ProgressCallback | None = None,
) -> AoTState:
    """Async convenience function."""
    graph = aot_graph(
        chatterer,
        max_depth=max_depth,
        max_sub_questions=max_sub_questions,
        max_workers=max_workers,
        on_progress=on_progress,
    )
    return await graph.ainvoke({"question": question})  # pyright: ignore[reportUnknownMemberType, reportReturnType]
