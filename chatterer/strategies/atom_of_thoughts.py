from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional, Type, TypeVar

from neo4j_extension import Graph, Neo4jConnection, Node, Relationship
from pydantic import BaseModel, Field, ValidationError

from ..language_model import Chatterer, LanguageModelInput
from .base import BaseStrategy

# ---------------------------------------------------------------------------------
# 0) Enums and Basic Models
# ---------------------------------------------------------------------------------


class Domain(StrEnum):
    """Defines the domain of a question for specialized handling."""

    GENERAL = "general"
    MATH = "math"
    CODING = "coding"
    PHILOSOPHY = "philosophy"
    MULTIHOP = "multihop"


class SubQuestionNode(BaseModel):
    """A single sub-question node in a decomposition tree."""

    question: str = Field(description="A sub-question string that arises from decomposition.")
    answer: Optional[str] = Field(description="Answer for this sub-question, if resolved.")
    depend: list[int] = Field(description="Indices of sub-questions that this node depends on.")


class RecursiveDecomposeResponse(BaseModel):
    """The result of a recursive decomposition step."""

    thought: str = Field(description="Reasoning about decomposition.")
    final_answer: str = Field(description="Best answer to the main question.")
    sub_questions: list[SubQuestionNode] = Field(description="Root-level sub-questions.")


class ContractQuestionResponse(BaseModel):
    """The result of contracting (simplifying) a question."""

    thought: str = Field(description="Reasoning on how the question was compressed.")
    question: str = Field(description="New, simplified, self-contained question.")


class EnsembleResponse(BaseModel):
    """The ensemble process result."""

    thought: str = Field(description="Explanation for choosing the final answer.")
    answer: str = Field(description="Best final answer after ensemble.")
    confidence: float = Field(description="Confidence score in [0, 1].")

    def model_post_init(self, __context: object) -> None:
        self.confidence = max(0.0, min(1.0, self.confidence))


class LabelResponse(BaseModel):
    thought: str = Field(description="Explanation or reasoning about labeling.")
    sub_questions: list[SubQuestionNode] = Field(
        description="Refined list of sub-questions with corrected dependencies."
    )


class CritiqueResponse(BaseModel):
    """A response used for LLM to self-critique or question its own correctness."""

    thought: str = Field(description="Critical reflection on correctness.")
    self_assessment: float = Field(description="Self-assessed confidence in the approach/answer. A float in [0,1].")


# ---------------------------------------------------------------------------------
# 1) Prompter Classes with Multi-Hop Context
# ---------------------------------------------------------------------------------


class BaseAoTPrompter(ABC):
    """Abstract base prompter that defines the required prompt methods."""

    @abstractmethod
    def recursive_decompose_prompt(
        self, question: str, sub_answers: Optional[str] = None, context: Optional[str] = None
    ) -> str: ...
    @abstractmethod
    def label_prompt(
        self, question: str, decompose_response: RecursiveDecomposeResponse, context: Optional[str] = None
    ) -> str: ...
    @abstractmethod
    def contract_prompt(self, question: str, sub_answers: str, context: Optional[str] = None) -> str: ...
    @abstractmethod
    def ensemble_prompt(
        self,
        original_question: str,
        decompose_answer: str,
        contracted_direct_answer: str,
        context: Optional[str] = None,
    ) -> str: ...
    @abstractmethod
    def critique_prompt(self, approach_name: str, answer: str, context: Optional[str] = None) -> str: ...


class GeneralAoTPrompter(BaseAoTPrompter):
    """Generic prompter for non-specialized or 'general' queries."""

    def recursive_decompose_prompt(
        self, question: str, sub_answers: Optional[str] = None, context: Optional[str] = None
    ) -> str:
        sub_ans_str = f"\nSub-question answers:\n{sub_answers}" if sub_answers else ""
        context_str = f"\nCONTEXT:\n{context}" if context else ""
        return (
            "You are a highly analytical assistant skilled in breaking down complex problems.\n"
            "Always think critically about whether your answer might be incorrect.\n"
            "Decompose the question into sub-questions recursively.\n\n"
            "REQUIREMENTS:\n"
            "1. Return valid JSON:\n"
            "   {\n"
            '     "thought": "...",\n'
            '     "final_answer": "...",\n'
            '     "sub_questions": [{"question": "...", "answer": null, "depend": []}, ...]\n'
            "   }\n"
            "2. 'thought': Provide detailed reasoning.\n"
            "3. 'final_answer': Integrate sub-answers if any.\n"
            "4. 'sub_questions': Key sub-questions with potential dependencies.\n\n"
            f"QUESTION:\n{question}{sub_ans_str}{context_str}"
        )

    def label_prompt(
        self, question: str, decompose_response: RecursiveDecomposeResponse, context: Optional[str] = None
    ) -> str:
        context_str = f"\nCONTEXT:\n{context}" if context else ""
        return (
            "You have a set of sub-questions from a decomposition process.\n"
            "We want to correct or refine the dependencies between sub-questions. Always check each step critically.\n\n"
            "REQUIREMENTS:\n"
            "1. Return valid JSON:\n"
            "   {\n"
            '     "thought": "...",\n'
            '     "sub_questions": [\n'
            '         {"question":"...", "answer":"...", "depend":[...]},\n'
            "         ...\n"
            "     ]\n"
            "   }\n"
            "2. 'thought': Provide reasoning about any changes.\n"
            "3. 'sub_questions': Possibly updated sub-questions with correct 'depend' lists.\n\n"
            f"ORIGINAL QUESTION:\n{question}\n"
            f"CURRENT DECOMPOSITION:\n{decompose_response.model_dump_json(indent=2)}"
            f"{context_str}"
        )

    def contract_prompt(self, question: str, sub_answers: str, context: Optional[str] = None) -> str:
        context_str = f"\nCONTEXT:\n{context}" if context else ""
        return (
            "You are tasked with compressing or simplifying a complex question into a single self-contained one.\n"
            "Please think carefully if there is anything contradictory or uncertain.\n\n"
            "REQUIREMENTS:\n"
            "1. Return valid JSON:\n"
            "   {'thought': '...', 'question': '...'}\n"
            "2. 'thought': Explain your simplification.\n"
            "3. 'question': The streamlined question.\n\n"
            f"ORIGINAL QUESTION:\n{question}\n"
            f"SUB-ANSWERS:\n{sub_answers}"
            f"{context_str}"
        )

    def ensemble_prompt(
        self,
        original_question: str,
        decompose_answer: str,
        contracted_direct_answer: str,
        context: Optional[str] = None,
    ) -> str:
        context_str = f"\nCONTEXT:\n{context}" if context else ""
        return (
            "You are an expert at synthesizing multiple candidate answers.\n"
            "Always check if any candidate might be wrong.\n"
            "Consider the following candidates:\n"
            f"1) Decomposition-based: {decompose_answer}\n"
            f"2) Contracted Direct: {contracted_direct_answer}\n\n"
            "REQUIREMENTS:\n"
            "1. Return valid JSON:\n"
            "   {'thought': '...', 'answer': '...', 'confidence': <float in [0,1]>}\n"
            "2. 'thought': Summarize how you decided.\n"
            "3. 'answer': Final best answer.\n"
            "4. 'confidence': Confidence score in [0,1].\n\n"
            f"ORIGINAL QUESTION:\n{original_question}"
            f"{context_str}"
        )

    def critique_prompt(self, approach_name: str, answer: str, context: Optional[str] = None) -> str:
        context_str = f"\nCONTEXT:\n{context}" if context else ""
        return (
            "You are asked to critique your own approach.\n"
            "Consider potential flaws or uncertainties in the solution.\n"
            f"APPROACH NAME: {approach_name}\n"
            f"ANSWER: {answer}\n"
            "Please provide a JSON with the following structure:\n"
            "{\n"
            '  "thought": "...",\n'
            '  "self_assessment": <float in [0,1]>\n'
            "}\n"
            "Where:\n"
            '- "thought" is a brief critical reflection\n'
            '- "self_assessment" is your estimated confidence in correctness\n'
            "Always remain suspicious of your own answers.\n"
            f"{context_str}"
        )


class MathAoTPrompter(GeneralAoTPrompter):
    """Specialized prompter for math questions; includes domain-specific hints."""

    def recursive_decompose_prompt(
        self, question: str, sub_answers: Optional[str] = None, context: Optional[str] = None
    ) -> str:
        base = super().recursive_decompose_prompt(question, sub_answers, context)
        return (
            base + "\nFocus on mathematical rigor and step-by-step derivations. Always question numeric correctness.\n"
        )


class CodingAoTPrompter(GeneralAoTPrompter):
    """Specialized prompter for coding/algorithmic queries."""

    def recursive_decompose_prompt(
        self, question: str, sub_answers: Optional[str] = None, context: Optional[str] = None
    ) -> str:
        base = super().recursive_decompose_prompt(question, sub_answers, context)
        return base + "\nBreak down into programming concepts or steps. Always double-check correctness.\n"


class PhilosophyAoTPrompter(GeneralAoTPrompter):
    """Specialized prompter for philosophical discussions."""

    def recursive_decompose_prompt(
        self, question: str, sub_answers: Optional[str] = None, context: Optional[str] = None
    ) -> str:
        base = super().recursive_decompose_prompt(question, sub_answers, context)
        return base + "\nConsider key philosophical theories and arguments. Be aware of contradictions.\n"


class MultiHopAoTPrompter(GeneralAoTPrompter):
    """Specialized prompter for multi-hop Q&A with explicit context usage."""

    def recursive_decompose_prompt(
        self, question: str, sub_answers: Optional[str] = None, context: Optional[str] = None
    ) -> str:
        base = super().recursive_decompose_prompt(question, sub_answers, context)
        return (
            base
            + "\nTreat this as a multi-hop question. Use the provided context carefully.\nExtract partial evidence from the context for each sub-question and verify each step.\n"
        )


# ---------------------------------------------------------------------------------
# 2) Strict Typed Steps for Pipeline
# ---------------------------------------------------------------------------------


class StepName(StrEnum):
    """Enum for step names in the pipeline."""

    DOMAIN_DETECTION = "DomainDetection"
    DECOMPOSITION = "Decomposition"
    DECOMPOSITION_CRITIQUE = "DecompositionCritique"
    CONTRACTED_QUESTION = "ContractedQuestion"
    CONTRACTED_DIRECT_ANSWER = "ContractedDirectAnswer"
    CONTRACT_CRITIQUE = "ContractCritique"
    BEST_APPROACH_DECISION = "BestApproachDecision"
    ENSEMBLE = "Ensemble"
    FINAL_ANSWER = "FinalAnswer"


class StepRelation(StrEnum):
    """Enum for relationship types in the reasoning graph."""

    CRITIQUES = "CRITIQUES"
    SELECTS = "SELECTS"
    RESULT_OF = "RESULT_OF"
    SPLIT_INTO = "SPLIT_INTO"
    DEPEND_ON = "DEPEND_ON"
    PRECEDES = "PRECEDES"
    DECOMPOSED_BY = "DECOMPOSED_BY"  # Added for recursive SubQuestion DAG


class StepRecord(BaseModel):
    """A typed record for each pipeline step."""

    step_name: StepName
    domain: Optional[str] = None
    score: Optional[float] = None
    used: Optional[StepName] = None
    sub_questions: Optional[list[SubQuestionNode]] = None
    parent_decomp_step_idx: Optional[int] = None
    parent_subq_idx: Optional[int] = None
    question: Optional[str] = None
    thought: Optional[str] = None
    answer: Optional[str] = None

    def as_properties(self) -> dict[str, str | float | int]:
        """Converts the StepRecord to a dictionary"""
        result: dict[str, str | float | int] = {}
        if self.score is not None:
            result["score"] = self.score
        if self.domain:
            result["domain"] = self.domain
        if self.question:
            result["question"] = self.question
        if self.thought:
            result["thought"] = self.thought
        if self.answer:
            result["answer"] = self.answer
        return result


# ---------------------------------------------------------------------------------
# 3) Logging Setup
# ---------------------------------------------------------------------------------


class SimpleColorFormatter(logging.Formatter):
    """Simple color-coded logging formatter for console output using ANSI escape codes."""

    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    LEVEL_COLORS = {
        logging.DEBUG: BLUE,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{log_color}{message}{self.RESET}"


logger = logging.getLogger("AoT")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(SimpleColorFormatter("%(levelname)s: %(message)s"))
logger.handlers = [handler]
logger.propagate = False

# ---------------------------------------------------------------------------------
# 4) The AoTPipeline Class (recursive + contract + ensemble)
# ---------------------------------------------------------------------------------

T = TypeVar(
    "T",
    bound=EnsembleResponse | ContractQuestionResponse | LabelResponse | CritiqueResponse | RecursiveDecomposeResponse,
)


@dataclass
class AoTPipeline:
    chatterer: Chatterer
    max_depth: int = 2
    max_retries: int = 2
    steps_history: list[StepRecord] = field(default_factory=list)
    prompter_map: dict[Domain, BaseAoTPrompter] = field(
        default_factory=lambda: {
            Domain.GENERAL: GeneralAoTPrompter(),
            Domain.MATH: MathAoTPrompter(),
            Domain.CODING: CodingAoTPrompter(),
            Domain.PHILOSOPHY: PhilosophyAoTPrompter(),
            Domain.MULTIHOP: MultiHopAoTPrompter(),
        }
    )

    def _record_decomposition_step(
        self,
        question: str,
        final_answer: str,
        sub_questions: list[SubQuestionNode],
        parent_decomp_step_idx: Optional[int] = None,
        parent_subq_idx: Optional[int] = None,
    ) -> int:
        """
        Save the results of a decomposition step to steps_history and return the index.
        """
        step_record = StepRecord(
            step_name=StepName.DECOMPOSITION,
            answer=final_answer,
            question=question,
            sub_questions=sub_questions,
            parent_decomp_step_idx=parent_decomp_step_idx,
            parent_subq_idx=parent_subq_idx,
        )
        self.steps_history.append(step_record)
        return len(self.steps_history) - 1

    async def _ainvoke_pydantic(
        self,
        prompt: str,
        model_cls: Type[T],
        fallback: str = "<None>",
    ) -> T:
        """Attempts up to max_retries to parse the model_cls from LLM output."""
        for attempt in range(1, self.max_retries + 1):
            try:
                return await self.chatterer.agenerate_pydantic(
                    response_model=model_cls, messages=[{"role": "user", "content": prompt}]
                )
            except ValidationError:
                if attempt == self.max_retries:
                    if issubclass(model_cls, EnsembleResponse):
                        return model_cls(thought=fallback, answer=fallback, confidence=0.0)
                    elif issubclass(model_cls, ContractQuestionResponse):
                        return model_cls(thought=fallback, question=fallback)
                    elif issubclass(model_cls, LabelResponse):
                        return model_cls(thought=fallback, sub_questions=[])
                    elif issubclass(model_cls, CritiqueResponse):
                        return model_cls(thought=fallback, self_assessment=0.0)
                    else:
                        return model_cls(thought=fallback, final_answer=fallback, sub_questions=[])
        raise RuntimeError("Unexpected error in _ainvoke_pydantic")

    async def _ainvoke_critique(
        self, approach_name: StepName, answer: str, prompter: BaseAoTPrompter, context: Optional[str] = None
    ) -> CritiqueResponse:
        critique_prompt = prompter.critique_prompt(approach_name, answer, context)
        return await self._ainvoke_pydantic(critique_prompt, CritiqueResponse)

    async def _adetect_domain(self, question: str, context: Optional[str]) -> Domain:
        class InferredDomain(BaseModel):
            domain: Domain

        ctx_str = f"\nCONTEXT:\n{context}" if context else ""
        domain_prompt = (
            "You are an expert domain classifier. "
            "Possible domains: [general, math, coding, philosophy, multihop].\n\n"
            "Return valid JSON: {'domain': '...'}. Always doubt your guess if uncertain.\n"
            f"QUESTION:\n{question}{ctx_str}"
        )
        try:
            result: InferredDomain = await self.chatterer.agenerate_pydantic(
                response_model=InferredDomain, messages=[{"role": "user", "content": domain_prompt}]
            )
            return result.domain
        except ValidationError:
            logger.warning("Failed domain detection, defaulting to general.")
            return Domain.GENERAL

    async def _arecursive_decompose_question(
        self,
        question: str,
        depth: int,
        prompter: BaseAoTPrompter,
        context: Optional[str],
        parent_decomp_step_idx: Optional[int] = None,
        parent_subq_idx: Optional[int] = None,
    ) -> RecursiveDecomposeResponse:
        """
        Recursively decomposes a question and records each step in steps_history.
        parent_decomp_step_idx / parent_subq_idx are used to link to the parent sub-question.
        """
        if depth < 0:
            return RecursiveDecomposeResponse(thought="Max depth reached", final_answer="Unknown", sub_questions=[])

        prompt: str = prompter.recursive_decompose_prompt(question, context=context)
        decompose_resp: RecursiveDecomposeResponse = await self._ainvoke_pydantic(prompt, RecursiveDecomposeResponse)

        # Labeling (refine dependencies)
        if decompose_resp.sub_questions:
            label_prompt: str = prompter.label_prompt(question, decompose_resp, context)
            label_resp: LabelResponse = await self._ainvoke_pydantic(label_prompt, LabelResponse)
            decompose_resp.sub_questions = label_resp.sub_questions

        current_decomp_step_idx: int = self._record_decomposition_step(
            question=question,
            final_answer=decompose_resp.final_answer,
            sub_questions=decompose_resp.sub_questions,
            parent_decomp_step_idx=parent_decomp_step_idx,
            parent_subq_idx=parent_subq_idx,
        )

        # Further resolution if depth remains
        if depth > 0 and decompose_resp.sub_questions:
            resolved_subs: list[SubQuestionNode] = await self._aresolve_sub_questions(
                decompose_resp.sub_questions, depth, prompter, context, current_decomp_step_idx
            )
            sub_answers_str: str = "\n".join(f"{sq.question}: {sq.answer}" for sq in resolved_subs if sq.answer)
            if sub_answers_str:
                refine_prompt: str = prompter.recursive_decompose_prompt(question, sub_answers_str, context)
                refined_resp: RecursiveDecomposeResponse = await self._ainvoke_pydantic(
                    refine_prompt, RecursiveDecomposeResponse
                )

                # update final_answer and sub_questions
                decompose_resp.final_answer = refined_resp.final_answer
                decompose_resp.sub_questions = resolved_subs

                # update steps_history
                self.steps_history[current_decomp_step_idx].answer = refined_resp.final_answer
                self.steps_history[current_decomp_step_idx].sub_questions = resolved_subs

        return decompose_resp

    async def _aresolve_sub_questions(
        self,
        sub_questions: list[SubQuestionNode],
        depth: int,
        prompter: BaseAoTPrompter,
        context: Optional[str],
        parent_decomp_step_idx: int,
    ) -> list[SubQuestionNode]:
        """Resolves sub-questions in topological order based on dependencies."""
        n = len(sub_questions)
        resolved: dict[int, SubQuestionNode] = {}
        in_degree: list[int] = [0] * n
        graph: list[list[int]] = [[] for _ in range(n)]
        for i, sq in enumerate(sub_questions):
            for dep in sq.depend:
                if 0 <= dep < n:
                    in_degree[i] += 1
                    graph[dep].append(i)

        queue: list[int] = [i for i in range(n) if in_degree[i] == 0]
        order: list[int] = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for nxt in graph[node]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        async def resolve_single_subq(idx: int) -> None:
            sq: SubQuestionNode = sub_questions[idx]
            sub_decomp: RecursiveDecomposeResponse = await self._arecursive_decompose_question(
                question=sq.question,
                depth=depth - 1,
                prompter=prompter,
                context=context,
                parent_decomp_step_idx=parent_decomp_step_idx,
                parent_subq_idx=idx,
            )
            sq.answer = sub_decomp.final_answer
            resolved[idx] = sq

        await asyncio.gather(*(resolve_single_subq(i) for i in order))
        return [resolved[i] for i in range(n) if i in resolved]

    def _calculate_score(self, answer: str, ground_truth: Optional[str], domain: Domain) -> float:
        """Scores an answer against ground truth; returns -1.0 if no ground truth."""
        if ground_truth is None:
            return -1.0
        if domain == Domain.MATH:
            try:
                ans_val = float(answer.strip())
                gt_val = float(ground_truth.strip())
                return 1.0 if abs(ans_val - gt_val) < 1e-9 else 0.0
            except ValueError:
                return 0.0
        return 1.0 if answer.strip().lower() == ground_truth.strip().lower() else 0.0

    async def arun_pipeline(
        self, question: str, context: Optional[str] = None, ground_truth: Optional[str] = None
    ) -> str:
        """Executes the full AoT pipeline, recording steps for graph construction."""
        self.steps_history.clear()

        # 1) Domain detection
        domain: Domain = await self._adetect_domain(question, context)
        self.steps_history.append(StepRecord(step_name=StepName.DOMAIN_DETECTION, domain=domain.value))
        prompter: BaseAoTPrompter = self.prompter_map[domain]
        logger.info(f"Detected domain: {domain}")

        # 2) Recursive Decomposition
        decomp_resp: RecursiveDecomposeResponse = await self._arecursive_decompose_question(
            question, self.max_depth, prompter, context
        )
        decompose_answer: str = decomp_resp.final_answer
        logger.info(f"Decomposition answer: {decompose_answer}")

        # 2.1) Self-critique
        decompose_critique: CritiqueResponse = await self._ainvoke_critique(
            StepName.DECOMPOSITION_CRITIQUE, decompose_answer, prompter, context
        )
        self.steps_history.append(
            StepRecord(
                step_name=StepName.DECOMPOSITION_CRITIQUE,
                score=decompose_critique.self_assessment,
                thought=decompose_critique.thought,
            )
        )
        decompose_actual_score = self._calculate_score(decompose_answer, ground_truth, domain)

        # 3) Contract question (optional step)
        sub_answers_str: str = ""
        top_level_decomp_steps: list[tuple[int, StepRecord]] = [
            (idx, s)
            for idx, s in enumerate(self.steps_history)
            if s.step_name == StepName.DECOMPOSITION and s.parent_decomp_step_idx is None
        ]
        if top_level_decomp_steps:
            # Though we could select a top-level step based on other criteria (e.g., score, depth),
            # here we simply use the last top-level decomposition step
            _, top_decomp_record = top_level_decomp_steps[-1]
            if top_decomp_record.sub_questions:
                sub_answers_str = "\n".join(
                    f"{sq.question}: {sq.answer}" for sq in top_decomp_record.sub_questions if sq.answer
                )

        contract_prompt: str = prompter.contract_prompt(question, sub_answers_str, context)
        contract_resp: ContractQuestionResponse = await self._ainvoke_pydantic(
            contract_prompt, ContractQuestionResponse
        )
        contracted_question: str = contract_resp.question
        self.steps_history.append(StepRecord(step_name=StepName.CONTRACTED_QUESTION, question=contracted_question))

        # 4) Direct approach on contracted question
        contracted_direct_prompt: str = (
            "You are an assistant refining the contracted question. Give a concise best-guess answer.\n\n"
            "REQUIREMENTS:\n"
            "1. JSON: {'thought': '...', 'answer': '...'}\n\n"
            f"CONTRACTED QUESTION: {contracted_question}"
        )
        contracted_direct_resp: RecursiveDecomposeResponse = await self._ainvoke_pydantic(
            contracted_direct_prompt,
            RecursiveDecomposeResponse,  # Used temporarily for format compatibility
            fallback="No Contracted Direct Answer",
        )
        contracted_direct_answer = contracted_direct_resp.final_answer
        self.steps_history.append(
            StepRecord(step_name=StepName.CONTRACTED_DIRECT_ANSWER, answer=contracted_direct_answer)
        )
        logger.info(f"Contracted direct answer: {contracted_direct_answer}")

        # 4.1) Self-critique
        contract_critique = await self._ainvoke_critique(
            StepName.CONTRACTED_DIRECT_ANSWER, contracted_direct_answer, prompter, context
        )
        self.steps_history.append(
            StepRecord(
                step_name=StepName.CONTRACT_CRITIQUE,
                score=contract_critique.self_assessment,
                thought=contract_critique.thought,
            )
        )
        contract_actual_score = self._calculate_score(contracted_direct_answer, ground_truth, domain)

        # 5) Compare approaches (decomp vs. contracted)
        decompose_final_score = (
            decompose_actual_score if decompose_actual_score >= 0 else decompose_critique.self_assessment
        )
        contract_final_score = (
            contract_actual_score if contract_actual_score >= 0 else contract_critique.self_assessment
        )

        best_score = max(decompose_final_score, contract_final_score)
        if best_score > 0 and best_score == decompose_final_score:
            best_approach_answer = decompose_answer
            approach_used = StepName.DECOMPOSITION
        elif best_score > 0 and best_score == contract_final_score:
            best_approach_answer = contracted_direct_answer
            approach_used = StepName.CONTRACTED_DIRECT_ANSWER
        else:
            ensemble_prompt = prompter.ensemble_prompt(
                original_question=question,
                decompose_answer=decompose_answer,
                contracted_direct_answer=contracted_direct_answer,
                context=context,
            )
            ensemble_resp: EnsembleResponse = await self._ainvoke_pydantic(ensemble_prompt, EnsembleResponse)
            best_approach_answer = ensemble_resp.answer
            approach_used = StepName.ENSEMBLE

        self.steps_history.append(StepRecord(step_name=StepName.BEST_APPROACH_DECISION, used=approach_used))
        logger.info(f"Choosing {approach_used} approach as best approach so far.")

        # 6) Final answer
        final_score = self._calculate_score(best_approach_answer, ground_truth, domain)
        self.steps_history.append(
            StepRecord(step_name=StepName.FINAL_ANSWER, answer=best_approach_answer, score=final_score)
        )
        logger.info(f"Final Answer: {best_approach_answer}")

        return best_approach_answer


# ---------------------------------------------------------------------------------
# 5) AoTStrategy with Graph Construction
# ---------------------------------------------------------------------------------


@dataclass
class AoTStrategy(BaseStrategy):
    """
    Strategy using AoTPipeline to process questions and provide a reasoning graph.
    The graph includes a single Decomposition node with a recursive DAG of SubQuestions.
    """

    pipeline: AoTPipeline

    async def ainvoke(self, messages: LanguageModelInput) -> str:
        """Asynchronously invokes the pipeline with the given messages."""
        input_ = self.pipeline.chatterer.client._convert_input(messages)  # type: ignore
        input_string = input_.to_string()
        return await self.pipeline.arun_pipeline(question=input_string)

    def invoke(self, messages: LanguageModelInput) -> str:
        """Synchronously invokes the pipeline with the given messages."""
        return asyncio.run(self.ainvoke(messages))

    def get_reasoning_graph(self, global_id_prefix: str = "AoT") -> Graph:
        """
        Constructs a Graph object from the pipeline's steps_history, capturing all reasoning steps.
        The Decomposition process is represented as a single node with a DAG of SubQuestions.

        Returns:
            Graph: A neo4j_extension Graph object with nodes and typed relationships.

        Relationships:
            - CRITIQUES: From critique steps to their targets.
            - SELECTS: From BestApproachDecision to the chosen approach.
            - RESULT_OF: From FinalAnswer to BestApproachDecision.
            - SPLIT_INTO: From the top-level Decomposition to its immediate SubQuestions.
            - DEPEND_ON: Between SubQuestions based on dependencies within the same level.
            - DECOMPOSED_BY: From a SubQuestion to its further decomposed SubQuestions.
            - PRECEDES: Between pipeline steps, excluding within the SubQuestion DAG.
        """
        g = Graph()
        step_nodes: dict[int, Node] = {}  # Indexed by step_history index
        subq_nodes: dict[str, Node] = {}  # Keyed by unique SubQuestion ID

        # Step 1: Create nodes for all steps except consolidating Decomposition
        for i, record in enumerate(self.pipeline.steps_history):
            if record.step_name == StepName.DECOMPOSITION and record.parent_decomp_step_idx is not None:
                continue  # Skip nested Decomposition steps; handle SubQuestions later
            step_node = Node(
                properties=record.as_properties(),
                labels={record.step_name},
                globalId=f"{global_id_prefix}_step_{i}",
            )
            g.add_node(step_node)
            step_nodes[i] = step_node

        # Step 2: Identify the top-level Decomposition and collect all SubQuestions
        top_decomp_idx = None
        all_sub_questions: dict[str, tuple[int, int, SubQuestionNode]] = {}
        for i, record in enumerate(self.pipeline.steps_history):
            if record.step_name == StepName.DECOMPOSITION:
                if record.parent_decomp_step_idx is None:
                    top_decomp_idx = i
                if record.sub_questions:
                    for sq_idx, sq in enumerate(record.sub_questions):
                        sq_id = f"{global_id_prefix}_decomp_{i}_sub_{sq_idx}"
                        all_sub_questions[sq_id] = (i, sq_idx, sq)

        if top_decomp_idx is None:
            logger.warning("No top-level Decomposition found; graph may be incomplete.")
            top_decomp_node = None
        else:
            top_decomp_node = step_nodes[top_decomp_idx]

        # Step 3: Create SubQuestion nodes
        for sq_id, (_, sq_idx, sq) in all_sub_questions.items():
            subq_node = Node(
                properties={
                    "question": sq.question,
                    "answer": sq.answer if sq.answer else "",
                },
                labels={"SubQuestion"},
                globalId=sq_id,
            )
            g.add_node(subq_node)
            subq_nodes[sq_id] = subq_node

        # Step 4: Add SPLIT_INTO from top-level Decomposition to its immediate SubQuestions
        if top_decomp_node and top_decomp_idx is not None:
            top_record = self.pipeline.steps_history[top_decomp_idx]
            for sq_idx, sq in enumerate(top_record.sub_questions or []):
                sq_id = f"{global_id_prefix}_decomp_{top_decomp_idx}_sub_{sq_idx}"
                if sq_id in subq_nodes:
                    g.add_relationship(
                        Relationship(
                            properties={},
                            rel_type=StepRelation.SPLIT_INTO,
                            start_node=top_decomp_node,
                            end_node=subq_nodes[sq_id],
                            globalId=f"{global_id_prefix}_split_{top_decomp_idx}_{sq_idx}",
                        )
                    )

        # Step 5: Add DECOMPOSED_BY relationships for recursive structure
        for i, record in enumerate(self.pipeline.steps_history):
            if (
                record.step_name == StepName.DECOMPOSITION
                and record.parent_decomp_step_idx is not None
                and record.parent_subq_idx is not None
            ):
                parent_decomp_idx = record.parent_decomp_step_idx
                parent_sq_idx = record.parent_subq_idx
                parent_sq_id = f"{global_id_prefix}_decomp_{parent_decomp_idx}_sub_{parent_sq_idx}"
                if parent_sq_id in subq_nodes:
                    parent_subq_node = subq_nodes[parent_sq_id]
                    for sq_idx, sq in enumerate(record.sub_questions or []):
                        child_sq_id = f"{global_id_prefix}_decomp_{i}_sub_{sq_idx}"
                        if child_sq_id in subq_nodes:
                            g.add_relationship(
                                Relationship(
                                    properties={},
                                    rel_type=StepRelation.DECOMPOSED_BY,
                                    start_node=parent_subq_node,
                                    end_node=subq_nodes[child_sq_id],
                                    globalId=f"{global_id_prefix}_decomposed_by_{i}_{sq_idx}",
                                )
                            )

        # Step 6: Add DEPEND_ON relationships within each Decomposition level
        for i, record in enumerate(self.pipeline.steps_history):
            if record.step_name == StepName.DECOMPOSITION and record.sub_questions:
                for sq_idx, sq in enumerate(record.sub_questions):
                    sub_q_id = f"{global_id_prefix}_decomp_{i}_sub_{sq_idx}"
                    if sub_q_id in subq_nodes:
                        sub_q_node = subq_nodes[sub_q_id]
                        for dep_idx in sq.depend or []:
                            dep_q_id = f"{global_id_prefix}_decomp_{i}_sub_{dep_idx}"
                            if dep_q_id in subq_nodes:
                                g.add_relationship(
                                    Relationship(
                                        properties={},
                                        rel_type=StepRelation.DEPEND_ON,
                                        start_node=sub_q_node,
                                        end_node=subq_nodes[dep_q_id],
                                        globalId=f"{global_id_prefix}_dep_{i}_{sq_idx}_on_{dep_idx}",
                                    )
                                )

        # Step 7: Add PRECEDES relationships, excluding within Decomposition DAG
        prev_non_decomp_idx = None
        for i, record in enumerate(self.pipeline.steps_history):
            if record.step_name == StepName.DECOMPOSITION and record.parent_decomp_step_idx is None:
                # Top-level Decomposition
                if prev_non_decomp_idx is not None:
                    g.add_relationship(
                        Relationship(
                            properties={},
                            rel_type=StepRelation.PRECEDES,
                            start_node=step_nodes[prev_non_decomp_idx],
                            end_node=step_nodes[i],
                            globalId=f"{global_id_prefix}_precede_{prev_non_decomp_idx}_to_{i}",
                        )
                    )
                prev_non_decomp_idx = i
            elif record.step_name != StepName.DECOMPOSITION:
                # Non-Decomposition steps
                if prev_non_decomp_idx is not None:
                    g.add_relationship(
                        Relationship(
                            properties={},
                            rel_type=StepRelation.PRECEDES,
                            start_node=step_nodes[prev_non_decomp_idx],
                            end_node=step_nodes[i],
                            globalId=f"{global_id_prefix}_precede_{prev_non_decomp_idx}_to_{i}",
                        )
                    )
                prev_non_decomp_idx = i

        # Step 8: Add CRITIQUES, SELECTS, and RESULT_OF relationships
        for i, record in enumerate(self.pipeline.steps_history):
            if i not in step_nodes:
                continue
            node = step_nodes[i]
            if record.step_name == StepName.DECOMPOSITION_CRITIQUE and top_decomp_idx is not None:
                g.add_relationship(
                    Relationship(
                        properties={"type": StepRelation.CRITIQUES},
                        rel_type=StepRelation.CRITIQUES,
                        start_node=node,
                        end_node=step_nodes[top_decomp_idx],
                        globalId=f"{global_id_prefix}_crit_decomp_{i}",
                    )
                )
            elif record.step_name == StepName.CONTRACT_CRITIQUE:
                for j in step_nodes:
                    if self.pipeline.steps_history[j].step_name == StepName.CONTRACTED_DIRECT_ANSWER:
                        g.add_relationship(
                            Relationship(
                                properties={"type": StepRelation.CRITIQUES},
                                rel_type=StepRelation.CRITIQUES,
                                start_node=node,
                                end_node=step_nodes[j],
                                globalId=f"{global_id_prefix}_crit_contract_{i}",
                            )
                        )

        best_decision_idx = None
        for i in step_nodes:
            record = self.pipeline.steps_history[i]
            if record.step_name == StepName.BEST_APPROACH_DECISION and record.used:
                best_decision_idx = i
                start_node = step_nodes[i]
                for j in step_nodes:
                    if self.pipeline.steps_history[j].step_name == record.used:
                        g.add_relationship(
                            Relationship(
                                properties={"decision": record.used},
                                rel_type=StepRelation.SELECTS,
                                start_node=start_node,
                                end_node=step_nodes[j],
                                globalId=f"{global_id_prefix}_decision_{i}",
                            )
                        )

        final_answer_idx = None
        for i in step_nodes:
            if self.pipeline.steps_history[i].step_name == StepName.FINAL_ANSWER:
                final_answer_idx = i
        if final_answer_idx is not None and best_decision_idx is not None:
            g.add_relationship(
                Relationship(
                    properties={},
                    rel_type=StepRelation.RESULT_OF,
                    start_node=step_nodes[final_answer_idx],
                    end_node=step_nodes[best_decision_idx],
                    globalId=f"{global_id_prefix}_final_result_{final_answer_idx}",
                )
            )

        return g


# ---------------------------------------------------------------------------------
# 6) Example Usage
# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    pipeline = AoTPipeline(chatterer=Chatterer.openai())
    strategy = AoTStrategy(pipeline=pipeline)
    question = "Which one is the larger number, 9.11 or 9.9?"
    answer = strategy.invoke(question)
    print("Final Answer:", answer)
    graph = strategy.get_reasoning_graph()
    print("\nGraph constructed with", len(graph.nodes), "nodes and", len(graph.relationships), "relationships.")
    with Neo4jConnection() as conn:
        conn.clear_all()
        conn.upsert_graph(graph)
        print("Graph stored in Neo4j database.")
