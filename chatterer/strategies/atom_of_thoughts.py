from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError

from ..language_model import Chatterer, LanguageModelInput
from ..messages import AIMessage, BaseMessage, HumanMessage
from .base import BaseStrategy

# ---------------------------------------------------------------------------------
# 0) Enums and Basic Models
# ---------------------------------------------------------------------------------

QA_TEMPLATE = "Q: {question}\nA: {answer}"
MAX_DEPTH_REACHED = "Max depth reached in recursive decomposition."
UNKNOWN = "Unknown"


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
    """Used to refine and reorder the sub-questions with corrected dependencies."""

    thought: str = Field(description="Explanation or reasoning about labeling.")
    sub_questions: list[SubQuestionNode] = Field(
        description="Refined list of sub-questions with corrected dependencies."
    )


class CritiqueResponse(BaseModel):
    """A response used for LLM to self-critique or question its own correctness."""

    thought: str = Field(description="Critical reflection on correctness.")
    self_assessment: float = Field(description="Self-assessed confidence in the approach/answer. A float in [0,1].")


# ---------------------------------------------------------------------------------
# [NEW] Additional classes to incorporate a separate sub-question devil's advocate
# ---------------------------------------------------------------------------------


class DevilsAdvocateResponse(BaseModel):
    """
    A response for a 'devil's advocate' pass.
    We consider an alternative viewpoint or contradictory answer.
    """

    thought: str = Field(description="Reasoning behind the contradictory viewpoint.")
    final_answer: str = Field(description="Alternative or conflicting answer to challenge the main one.")
    sub_questions: list[SubQuestionNode] = Field(
        description="Any additional sub-questions from the contrarian perspective."
    )


# ---------------------------------------------------------------------------------
# 1) Prompter Classes with Multi-Hop + Devil's Advocate
# ---------------------------------------------------------------------------------


class AoTPrompter:
    """Generic base prompter that defines the required prompt methods."""

    def recursive_decompose_prompt(
        self, messages: list[BaseMessage], question: str, sub_answers: list[tuple[str, str]]
    ) -> list[BaseMessage]:
        """
        Prompt for main decomposition.
        Encourages step-by-step reasoning and listing sub-questions as JSON.
        """
        decompose_instructions = (
            "First, restate the main question.\n"
            "Decide if sub-questions are needed. If so, list them.\n"
            "In the 'thought' field, show your chain-of-thought.\n"
            "Return valid JSON:\n"
            "{\n"
            '  "thought": "...",\n'
            '  "final_answer": "...",\n'
            '  "sub_questions": [\n'
            '    {"question": "...", "answer": null, "depend": []},\n'
            "    ...\n"
            "  ]\n"
            "}\n"
        )

        content_sub_answers = "\n".join(f"Sub-answer so far: Q={q}, A={a}" for q, a in sub_answers)
        return messages + [
            HumanMessage(content=f"Main question:\n{question}"),
            AIMessage(content=content_sub_answers),
            AIMessage(content=decompose_instructions),
        ]

    def label_prompt(
        self, messages: list[BaseMessage], question: str, decompose_response: RecursiveDecomposeResponse
    ) -> list[BaseMessage]:
        """
        Prompt for refining the sub-questions and dependencies.
        """
        label_instructions = (
            "Review each sub-question to ensure correctness and proper ordering.\n"
            "Return valid JSON in the form:\n"
            "{\n"
            '  "thought": "...",\n'
            '  "sub_questions": [\n'
            '    {"question": "...", "answer": "...", "depend": [...]},\n'
            "    ...\n"
            "  ]\n"
            "}\n"
        )
        return messages + [
            AIMessage(content=f"Question: {question}"),
            AIMessage(content=f"Current sub-questions:\n{decompose_response.sub_questions}"),
            AIMessage(content=label_instructions),
        ]

    def contract_prompt(self, messages: list[BaseMessage], sub_answers: list[tuple[str, str]]) -> list[BaseMessage]:
        """
        Prompt for merging sub-answers into one self-contained question.
        """
        contract_instructions = (
            "Please merge sub-answers into a single short question that is fully self-contained.\n"
            "In 'thought', show how you unify the information.\n"
            "Then produce JSON:\n"
            "{\n"
            '  "thought": "...",\n'
            '  "question": "a short but self-contained question"\n'
            "}\n"
        )
        sub_q_content = "\n".join(f"Q: {q}\nA: {a}" for q, a in sub_answers)
        return messages + [
            AIMessage(content="We have these sub-questions and answers:"),
            AIMessage(content=sub_q_content),
            AIMessage(content=contract_instructions),
        ]

    def contract_direct_prompt(self, messages: list[BaseMessage], contracted_question: str) -> list[BaseMessage]:
        """
        Prompt for directly answering the contracted question thoroughly.
        """
        direct_instructions = (
            "Answer the simplified question thoroughly. Show your chain-of-thought in 'thought'.\n"
            "Return JSON:\n"
            "{\n"
            '  "thought": "...",\n'
            '  "final_answer": "..."\n'
            "}\n"
        )
        return messages + [
            HumanMessage(content=f"Simplified question: {contracted_question}"),
            AIMessage(content=direct_instructions),
        ]

    def critique_prompt(self, messages: list[BaseMessage], thought: str, answer: str) -> list[BaseMessage]:
        """
        Prompt for self-critique.
        """
        critique_instructions = (
            "Critique your own approach. Identify possible errors or leaps.\n"
            "Return JSON:\n"
            "{\n"
            '  "thought": "...",\n'
            '  "self_assessment": <float in [0,1]>\n'
            "}\n"
        )
        return messages + [
            AIMessage(content=f"Your previous THOUGHT:\n{thought}"),
            AIMessage(content=f"Your previous ANSWER:\n{answer}"),
            AIMessage(content=critique_instructions),
        ]

    def ensemble_prompt(
        self, messages: list[BaseMessage], possible_thought_and_answers: list[tuple[str, str]]
    ) -> list[BaseMessage]:
        """
        Show multiple candidate solutions and pick the best final answer with confidence.
        """
        instructions = (
            "You have multiple candidate solutions. Compare carefully and pick the best.\n"
            "Return JSON:\n"
            "{\n"
            '  "thought": "why you chose this final answer",\n'
            '  "answer": "the best consolidated answer",\n'
            '  "confidence": 0.0 ~ 1.0\n'
            "}\n"
        )
        reasonings: list[BaseMessage] = []
        for idx, (thought, ans) in enumerate(possible_thought_and_answers):
            reasonings.append(AIMessage(content=f"[Candidate {idx}] Thought:\n{thought}\nAnswer:\n{ans}\n---"))
        return messages + reasonings + [AIMessage(content=instructions)]

    def devils_advocate_prompt(
        self, messages: list[BaseMessage], question: str, existing_answer: str
    ) -> list[BaseMessage]:
        """
        Prompt for a devil's advocate approach to contradict or provide an alternative viewpoint.
        """
        instructions = (
            "Act as a devil's advocate. Suppose the existing answer is incomplete or incorrect.\n"
            "Challenge it, find alternative ways or details. Provide a new 'final_answer' (even if contradictory).\n"
            "Return JSON in the same shape as RecursiveDecomposeResponse OR a dedicated structure.\n"
            "But here, let's keep it in a new dedicated structure:\n"
            "{\n"
            '  "thought": "...",\n'
            '  "final_answer": "...",\n'
            '  "sub_questions": [\n'
            '    {"question": "...", "answer": null, "depend": []},\n'
            "    ...\n"
            "  ]\n"
            "}\n"
        )
        return messages + [
            AIMessage(content=(f"Current question: {question}\nExisting answer to challenge: {existing_answer}\n")),
            AIMessage(content=instructions),
        ]


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

    DEVILS_ADVOCATE = "DevilsAdvocate"
    DEVILS_ADVOCATE_CRITIQUE = "DevilsAdvocateCritique"


class StepRelation(StrEnum):
    """Enum for relationship types in the reasoning graph."""

    CRITIQUES = "CRITIQUES"
    SELECTS = "SELECTS"
    RESULT_OF = "RESULT_OF"
    SPLIT_INTO = "SPLIT_INTO"
    DEPEND_ON = "DEPEND_ON"
    PRECEDES = "PRECEDES"
    DECOMPOSED_BY = "DECOMPOSED_BY"


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

    def as_properties(self) -> dict[str, str | float | int | None]:
        """Converts the StepRecord to a dictionary of properties."""
        result: dict[str, str | float | int | None] = {}
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
# 4) The AoTPipeline Class (now with recursive devil's advocate at each sub-question)
# ---------------------------------------------------------------------------------

T = TypeVar(
    "T",
    bound=EnsembleResponse
    | ContractQuestionResponse
    | LabelResponse
    | CritiqueResponse
    | RecursiveDecomposeResponse
    | DevilsAdvocateResponse,
)


@dataclass
class AoTPipeline:
    """
    The pipeline orchestrates:
      1) Recursive decomposition
      2) For each sub-question, it tries a main approach + a devil's advocate approach
      3) Merges sub-answers using an ensemble
      4) Contracts the question
      5) Possibly does a direct approach on the contracted question
      6) Ensembling the final answers
    """

    chatterer: Chatterer
    max_depth: int = 2
    max_retries: int = 2
    steps_history: list[StepRecord] = field(default_factory=list)
    prompter: AoTPrompter = field(default_factory=AoTPrompter)

    # 4.1) Utility for calling the LLM with Pydantic parsing
    async def _ainvoke_pydantic(
        self,
        messages: list[BaseMessage],
        model_cls: Type[T],
        fallback: str = "<None>",
    ) -> T:
        """
        Attempts up to max_retries to parse the model_cls from LLM output as JSON.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                return await self.chatterer.agenerate_pydantic(response_model=model_cls, messages=messages)
            except ValidationError as e:
                logger.warning(f"ValidationError on attempt {attempt} for {model_cls.__name__}: {e}")
                if attempt == self.max_retries:
                    # Return a fallback version
                    if issubclass(model_cls, EnsembleResponse):
                        return model_cls(thought=fallback, answer=fallback, confidence=0.0)  # type: ignore
                    elif issubclass(model_cls, ContractQuestionResponse):
                        return model_cls(thought=fallback, question=fallback)  # type: ignore
                    elif issubclass(model_cls, LabelResponse):
                        return model_cls(thought=fallback, sub_questions=[])  # type: ignore
                    elif issubclass(model_cls, CritiqueResponse):
                        return model_cls(thought=fallback, self_assessment=0.0)  # type: ignore
                    elif issubclass(model_cls, DevilsAdvocateResponse):
                        return model_cls(thought=fallback, final_answer=fallback, sub_questions=[])  # type: ignore
                    else:
                        return model_cls(thought=fallback, final_answer=fallback, sub_questions=[])  # type: ignore
        # theoretically unreachable
        raise RuntimeError("Unexpected error in _ainvoke_pydantic")

    # 4.2) Helper method for self-critique
    async def _ainvoke_critique(
        self,
        messages: list[BaseMessage],
        thought: str,
        answer: str,
    ) -> CritiqueResponse:
        """
        Instructs the LLM to critique the given thought & answer, returning CritiqueResponse.
        """
        return await self._ainvoke_pydantic(
            messages=self.prompter.critique_prompt(messages=messages, thought=thought, answer=answer),
            model_cls=CritiqueResponse,
        )

    # 4.3) Helper method for devil's advocate approach
    async def _ainvoke_devils_advocate(
        self,
        messages: list[BaseMessage],
        question: str,
        existing_answer: str,
    ) -> DevilsAdvocateResponse:
        """
        Instructs the LLM to challenge an existing answer with a devil's advocate approach.
        """
        return await self._ainvoke_pydantic(
            messages=self.prompter.devils_advocate_prompt(messages, question=question, existing_answer=existing_answer),
            model_cls=DevilsAdvocateResponse,
        )

    # 4.4) The main function that recursively decomposes a question and calls sub-steps
    async def _arecursive_decompose_question(
        self,
        messages: list[BaseMessage],
        question: str,
        depth: int,
        parent_decomp_step_idx: Optional[int] = None,
        parent_subq_idx: Optional[int] = None,
    ) -> RecursiveDecomposeResponse:
        """
        Recursively decompose the given question. For each sub-question:
          1) Recursively decompose that sub-question if we still have depth left
          2) After getting a main sub-answer, do a devil's advocate pass
          3) Combine main sub-answer + devil's advocate alternative via an ensemble
        """
        if depth < 0:
            logger.info("Max depth reached, returning unknown.")
            return RecursiveDecomposeResponse(thought=MAX_DEPTH_REACHED, final_answer=UNKNOWN, sub_questions=[])

        # Step 1: Perform the decomposition
        decompose_resp: RecursiveDecomposeResponse = await self._ainvoke_pydantic(
            messages=self.prompter.recursive_decompose_prompt(messages=messages, question=question, sub_answers=[]),
            model_cls=RecursiveDecomposeResponse,
        )

        # Step 2: Label / refine sub-questions (dependencies, ordering)
        if decompose_resp.sub_questions:
            label_resp: LabelResponse = await self._ainvoke_pydantic(
                messages=self.prompter.label_prompt(messages, question, decompose_resp),
                model_cls=LabelResponse,
            )
            decompose_resp.sub_questions = label_resp.sub_questions

        # Save a pipeline record for this decomposition step
        current_decomp_step_idx = self._record_decomposition_step(
            question=question,
            final_answer=decompose_resp.final_answer,
            sub_questions=decompose_resp.sub_questions,
            parent_decomp_step_idx=parent_decomp_step_idx,
            parent_subq_idx=parent_subq_idx,
        )

        # Step 3: If sub-questions exist and depth remains, solve them + do devil's advocate
        if depth > 0 and decompose_resp.sub_questions:
            solved_subs: list[SubQuestionNode] = await self._aresolve_sub_questions(
                messages=messages,
                sub_questions=decompose_resp.sub_questions,
                depth=depth,
                parent_decomp_step_idx=current_decomp_step_idx,
            )
            # Then we can refine the "final_answer" from those sub-answers
            # or we do a secondary pass to refine the final answer
            refined_prompt = self.prompter.recursive_decompose_prompt(
                messages=messages,
                question=question,
                sub_answers=[(sq.question, sq.answer or UNKNOWN) for sq in solved_subs],
            )
            refined_resp: RecursiveDecomposeResponse = await self._ainvoke_pydantic(
                refined_prompt, RecursiveDecomposeResponse
            )
            decompose_resp.final_answer = refined_resp.final_answer
            decompose_resp.sub_questions = solved_subs

            # Update pipeline record
            self.steps_history[current_decomp_step_idx].answer = refined_resp.final_answer
            self.steps_history[current_decomp_step_idx].sub_questions = solved_subs

        return decompose_resp

    def _record_decomposition_step(
        self,
        question: str,
        final_answer: str,
        sub_questions: list[SubQuestionNode],
        parent_decomp_step_idx: Optional[int],
        parent_subq_idx: Optional[int],
    ) -> int:
        """
        Save the decomposition step in steps_history, returning the index.
        """
        step_record = StepRecord(
            step_name=StepName.DECOMPOSITION,
            question=question,
            answer=final_answer,
            sub_questions=sub_questions,
            parent_decomp_step_idx=parent_decomp_step_idx,
            parent_subq_idx=parent_subq_idx,
        )
        self.steps_history.append(step_record)
        return len(self.steps_history) - 1

    async def _aresolve_sub_questions(
        self,
        messages: list[BaseMessage],
        sub_questions: list[SubQuestionNode],
        depth: int,
        parent_decomp_step_idx: Optional[int],
    ) -> list[SubQuestionNode]:
        """
        Resolve sub-questions in topological order.
        For each sub-question:
          1) Recursively decompose (main approach).
          2) Acquire a devil's advocate alternative.
          3) Critique or ensemble if needed.
          4) Finalize sub-question answer.
        """
        n = len(sub_questions)
        in_degree = [0] * n
        graph: list[list[int]] = [[] for _ in range(n)]
        for i, sq in enumerate(sub_questions):
            for dep in sq.depend:
                if 0 <= dep < n:
                    in_degree[i] += 1
                    graph[dep].append(i)

        # Kahn's algorithm for topological order
        queue = [i for i in range(n) if in_degree[i] == 0]
        topo_order: list[int] = []

        while queue:
            node = queue.pop(0)
            topo_order.append(node)
            for nxt in graph[node]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        # We'll store the resolved sub-questions
        final_subs: dict[int, SubQuestionNode] = {}

        async def _resolve_one_subq(idx: int):
            sq = sub_questions[idx]
            # 1) Main approach
            main_resp = await self._arecursive_decompose_question(
                messages=messages,
                question=sq.question,
                depth=depth - 1,
                parent_decomp_step_idx=parent_decomp_step_idx,
                parent_subq_idx=idx,
            )

            main_answer = main_resp.final_answer

            # 2) Devil's Advocate approach
            devils_resp = await self._ainvoke_devils_advocate(
                messages=messages, question=sq.question, existing_answer=main_answer
            )
            # 3) Ensemble to combine main_answer + devils_alternative
            ensemble_sub = await self._ainvoke_pydantic(
                self.prompter.ensemble_prompt(
                    messages=messages,
                    possible_thought_and_answers=[
                        (main_resp.thought, main_answer),
                        (devils_resp.thought, devils_resp.final_answer),
                    ],
                ),
                EnsembleResponse,
            )
            sub_best_answer = ensemble_sub.answer

            # Store final subq answer
            sq.answer = sub_best_answer
            final_subs[idx] = sq

            # Record pipeline steps for devil's advocate
            self.steps_history.append(
                StepRecord(
                    step_name=StepName.DEVILS_ADVOCATE,
                    question=sq.question,
                    answer=devils_resp.final_answer,
                    thought=devils_resp.thought,
                    sub_questions=devils_resp.sub_questions,
                )
            )
            # Possibly critique the devils advocate result
            dev_adv_crit = await self._ainvoke_critique(
                messages=messages, thought=devils_resp.thought, answer=devils_resp.final_answer
            )
            self.steps_history.append(
                StepRecord(
                    step_name=StepName.DEVILS_ADVOCATE_CRITIQUE,
                    thought=dev_adv_crit.thought,
                    score=dev_adv_crit.self_assessment,
                )
            )

        # Solve sub-questions in topological order
        tasks = [_resolve_one_subq(i) for i in topo_order]
        await asyncio.gather(*tasks, return_exceptions=False)

        return [final_subs[i] for i in range(n)]

    # 4.5) The primary pipeline method
    async def arun_pipeline(self, messages: list[BaseMessage]) -> str:
        """
        Execute the pipeline:
          1) Decompose the main question (recursively).
          2) Self-critique.
          3) Provide a devil's advocate approach on the entire main result.
          4) Contract sub-answers (optional).
          5) Directly solve the contracted question.
          6) Self-critique again.
          7) Final ensemble across main vs devil's vs contracted direct answer.
          8) Return final answer.
        """
        self.steps_history.clear()

        original_question: str = messages[-1].text()
        # 1) Recursive decomposition
        decomp_resp = await self._arecursive_decompose_question(
            messages=messages,
            question=original_question,
            depth=self.max_depth,
        )
        logger.info(f"[Main Decomposition] final_answer={decomp_resp.final_answer}")

        # 2) Self-critique of main decomposition
        decomp_critique = await self._ainvoke_critique(
            messages=messages, thought=decomp_resp.thought, answer=decomp_resp.final_answer
        )
        self.steps_history.append(
            StepRecord(
                step_name=StepName.DECOMPOSITION_CRITIQUE,
                thought=decomp_critique.thought,
                score=decomp_critique.self_assessment,
            )
        )

        # 3) Devil's advocate on the entire main answer
        devils_on_main = await self._ainvoke_devils_advocate(
            messages=messages, question=original_question, existing_answer=decomp_resp.final_answer
        )
        self.steps_history.append(
            StepRecord(
                step_name=StepName.DEVILS_ADVOCATE,
                question=original_question,
                answer=devils_on_main.final_answer,
                thought=devils_on_main.thought,
                sub_questions=devils_on_main.sub_questions,
            )
        )
        devils_crit_main = await self._ainvoke_critique(
            messages=messages, thought=devils_on_main.thought, answer=devils_on_main.final_answer
        )
        self.steps_history.append(
            StepRecord(
                step_name=StepName.DEVILS_ADVOCATE_CRITIQUE,
                thought=devils_crit_main.thought,
                score=devils_crit_main.self_assessment,
            )
        )

        # 4) Contract sub-answers from main decomposition
        top_decomp_record: Optional[StepRecord] = next(
            (
                s
                for s in reversed(self.steps_history)
                if s.step_name == StepName.DECOMPOSITION and s.parent_decomp_step_idx is None
            ),
            None,
        )
        if top_decomp_record and top_decomp_record.sub_questions:
            sub_answers = [(sq.question, sq.answer or UNKNOWN) for sq in top_decomp_record.sub_questions]
        else:
            sub_answers = []

        contract_resp = await self._ainvoke_pydantic(
            messages=self.prompter.contract_prompt(messages, sub_answers),
            model_cls=ContractQuestionResponse,
        )
        contracted_question = contract_resp.question
        self.steps_history.append(
            StepRecord(
                step_name=StepName.CONTRACTED_QUESTION, question=contracted_question, thought=contract_resp.thought
            )
        )

        # 5) Attempt direct approach on contracted question
        contracted_direct = await self._ainvoke_pydantic(
            messages=self.prompter.contract_direct_prompt(messages, contracted_question),
            model_cls=RecursiveDecomposeResponse,
            fallback="No Contracted Direct Answer",
        )
        self.steps_history.append(
            StepRecord(
                step_name=StepName.CONTRACTED_DIRECT_ANSWER,
                answer=contracted_direct.final_answer,
                thought=contracted_direct.thought,
            )
        )
        logger.info(f"[Contracted Direct] final_answer={contracted_direct.final_answer}")

        # 5.1) Critique the contracted direct approach
        contract_critique = await self._ainvoke_critique(
            messages=messages, thought=contracted_direct.thought, answer=contracted_direct.final_answer
        )
        self.steps_history.append(
            StepRecord(
                step_name=StepName.CONTRACT_CRITIQUE,
                thought=contract_critique.thought,
                score=contract_critique.self_assessment,
            )
        )

        # 6) Ensemble of (Main decomposition, Devil's advocate on main, Contracted direct)
        ensemble_resp = await self._ainvoke_pydantic(
            self.prompter.ensemble_prompt(
                messages=messages,
                possible_thought_and_answers=[
                    (decomp_resp.thought, decomp_resp.final_answer),
                    (devils_on_main.thought, devils_on_main.final_answer),
                    (contracted_direct.thought, contracted_direct.final_answer),
                ],
            ),
            EnsembleResponse,
        )
        best_approach_answer = ensemble_resp.answer
        approach_used = StepName.ENSEMBLE
        self.steps_history.append(StepRecord(step_name=StepName.BEST_APPROACH_DECISION, used=approach_used))
        logger.info(f"[Best Approach Decision] => {approach_used}")

        # 7) Final answer
        self.steps_history.append(
            StepRecord(step_name=StepName.FINAL_ANSWER, answer=best_approach_answer, score=ensemble_resp.confidence)
        )
        logger.info(f"[Final Answer] => {best_approach_answer}")

        return best_approach_answer

    def run_pipeline(self, messages: list[BaseMessage]) -> str:
        """Synchronous wrapper around arun_pipeline."""
        return asyncio.run(self.arun_pipeline(messages))

    # ---------------------------------------------------------------------------------
    # 4.6) Build or export a reasoning graph
    # ---------------------------------------------------------------------------------

    def get_reasoning_graph(self, global_id_prefix: str = "AoT"):
        """
        Constructs a Graph object (from hypothetical `neo4j_extension`)
        capturing the pipeline steps, including devil's advocate steps.
        """
        from neo4j_extension import Graph, Node, Relationship

        g = Graph()
        step_nodes: dict[int, Node] = {}
        subq_nodes: dict[str, Node] = {}

        # Step A: Create nodes for each pipeline step
        for i, record in enumerate(self.steps_history):
            # We'll skip nested Decomposition steps only if we want to flatten them.
            # But let's keep them for clarity.
            step_node = Node(
                properties=record.as_properties(), labels={record.step_name}, globalId=f"{global_id_prefix}_step_{i}"
            )
            g.add_node(step_node)
            step_nodes[i] = step_node

        # Step B: Collect sub-questions from each DECOMPOSITION or DEVILS_ADVOCATE
        all_sub_questions: dict[str, tuple[int, int, SubQuestionNode]] = {}
        for i, record in enumerate(self.steps_history):
            if record.sub_questions:
                for sq_idx, sq in enumerate(record.sub_questions):
                    sq_id = f"{global_id_prefix}_decomp_{i}_sub_{sq_idx}"
                    all_sub_questions[sq_id] = (i, sq_idx, sq)

        for sq_id, (i, sq_idx, sq) in all_sub_questions.items():
            n_subq = Node(
                properties={
                    "question": sq.question,
                    "answer": sq.answer or "",
                },
                labels={"SubQuestion"},
                globalId=sq_id,
            )
            g.add_node(n_subq)
            subq_nodes[sq_id] = n_subq

        # Step C: Add relationships. We do a simple approach:
        #  - If StepRecord is DECOMPOSITION or DEVILS_ADVOCATE with sub_questions, link them via SPLIT_INTO.
        for i, record in enumerate(self.steps_history):
            if record.sub_questions:
                start_node = step_nodes[i]
                for sq_idx, sq in enumerate(record.sub_questions):
                    sq_id = f"{global_id_prefix}_decomp_{i}_sub_{sq_idx}"
                    end_node = subq_nodes[sq_id]
                    rel = Relationship(
                        properties={},
                        rel_type=StepRelation.SPLIT_INTO,
                        start_node=start_node,
                        end_node=end_node,
                        globalId=f"{global_id_prefix}_split_{i}_{sq_idx}",
                    )
                    g.add_relationship(rel)
                    # Also add sub-question dependencies
                    for dep in sq.depend:
                        # The same record i -> sub-question subq
                        if 0 <= dep < len(record.sub_questions):
                            dep_id = f"{global_id_prefix}_decomp_{i}_sub_{dep}"
                            if dep_id in subq_nodes:
                                dep_node = subq_nodes[dep_id]
                                rel_dep = Relationship(
                                    properties={},
                                    rel_type=StepRelation.DEPEND_ON,
                                    start_node=end_node,
                                    end_node=dep_node,
                                    globalId=f"{global_id_prefix}_dep_{i}_q_{sq_idx}_on_{dep}",
                                )
                                g.add_relationship(rel_dep)

        # Step D: We add PRECEDES relationships in a linear chain for the pipeline steps
        for i in range(len(self.steps_history) - 1):
            start_node = step_nodes[i]
            end_node = step_nodes[i + 1]
            rel = Relationship(
                properties={},
                rel_type=StepRelation.PRECEDES,
                start_node=start_node,
                end_node=end_node,
                globalId=f"{global_id_prefix}_precede_{i}_to_{i + 1}",
            )
            g.add_relationship(rel)

        # Step E: CRITIQUES, SELECTS, RESULT_OF can be similarly added:
        # We'll do a simple pass:
        # If step_name ends with CRITIQUE => it critiques the step before it
        for i, record in enumerate(self.steps_history):
            if "CRITIQUE" in record.step_name:
                # Let it point to the preceding step
                if i > 0:
                    start_node = step_nodes[i]
                    end_node = step_nodes[i - 1]
                    rel = Relationship(
                        properties={},
                        rel_type=StepRelation.CRITIQUES,
                        start_node=start_node,
                        end_node=end_node,
                        globalId=f"{global_id_prefix}_crit_{i}",
                    )
                    g.add_relationship(rel)

        # If there's a BEST_APPROACH_DECISION step, link it to the step it uses
        best_decision_idx = None
        used_step_idx = None
        for i, record in enumerate(self.steps_history):
            if record.step_name == StepName.BEST_APPROACH_DECISION and record.used:
                best_decision_idx = i
                # find the step with that name
                used_step_idx = next((j for j in step_nodes if self.steps_history[j].step_name == record.used), None)
                if used_step_idx is not None:
                    rel = Relationship(
                        properties={},
                        rel_type=StepRelation.SELECTS,
                        start_node=step_nodes[i],
                        end_node=step_nodes[used_step_idx],
                        globalId=f"{global_id_prefix}_select_{i}_use_{used_step_idx}",
                    )
                    g.add_relationship(rel)

        # And link the final answer to the best approach
        final_answer_idx = next(
            (i for i, r in enumerate(self.steps_history) if r.step_name == StepName.FINAL_ANSWER), None
        )
        if final_answer_idx is not None and best_decision_idx is not None:
            rel = Relationship(
                properties={},
                rel_type=StepRelation.RESULT_OF,
                start_node=step_nodes[final_answer_idx],
                end_node=step_nodes[best_decision_idx],
                globalId=f"{global_id_prefix}_final_{final_answer_idx}_resultof_{best_decision_idx}",
            )
            g.add_relationship(rel)

        return g


# ---------------------------------------------------------------------------------
# 5) AoTStrategy class that uses the pipeline
# ---------------------------------------------------------------------------------


@dataclass
class AoTStrategy(BaseStrategy):
    """
    Strategy using AoTPipeline with a reasoning graph and deeper devil's advocate.
    """

    pipeline: AoTPipeline

    async def ainvoke(self, messages: LanguageModelInput) -> str:
        """Asynchronously run the pipeline with the given messages."""
        # Convert your custom input to list[BaseMessage] as needed:
        msgs = self.pipeline.chatterer.client._convert_input(messages).to_messages()  # type: ignore
        return await self.pipeline.arun_pipeline(msgs)

    def invoke(self, messages: LanguageModelInput) -> str:
        """Synchronously run the pipeline with the given messages."""
        msgs = self.pipeline.chatterer.client._convert_input(messages).to_messages()  # type: ignore
        return self.pipeline.run_pipeline(msgs)

    def get_reasoning_graph(self):
        """Return the AoT reasoning graph from the pipeline’s steps history."""
        return self.pipeline.get_reasoning_graph(global_id_prefix="AoT")


# ---------------------------------------------------------------------------------
# Example usage (pseudo-code)
# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    from neo4j_extension import Neo4jConnection  # or your actual DB connector

    # You would create a Chatterer with your chosen LLM backend (OpenAI, etc.)
    chatterer = Chatterer.openai()  # pseudo-code
    pipeline = AoTPipeline(chatterer=chatterer, max_depth=3)
    strategy = AoTStrategy(pipeline=pipeline)

    question = "Solve 5.9 = 5.11 - x. Also compare 9.11 and 9.9."
    answer = strategy.invoke(question)
    print("Final Answer:", answer)

    # Build the reasoning graph
    graph = strategy.get_reasoning_graph()
    print(f"\nGraph has {len(graph.nodes)} nodes and {len(graph.relationships)} relationships.")

    # Optionally store in Neo4j
    with Neo4jConnection() as conn:
        conn.clear_all()
        conn.upsert_graph(graph)
        print("Graph stored in Neo4j.")
