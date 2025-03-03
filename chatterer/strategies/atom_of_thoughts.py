# original source: https://github.com/qixucen/atom

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import LiteralString, Optional, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError

from ..language_model import Chatterer, LanguageModelInput
from .base import BaseStrategy

logger = logging.getLogger(__name__)

# ------------------------- 0) Enums and Basic Models -------------------------


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
    depend: list[int] = Field(default_factory=list, description="Indices of sub-questions that this node depends on.")


class RecursiveDecomposeResponse(BaseModel):
    """The result of a recursive decomposition step."""

    thought: str = Field(description="Reasoning about decomposition.")
    final_answer: str = Field(description="Best answer to the main question.")
    sub_questions: list[SubQuestionNode] = Field(default_factory=list, description="Root-level sub-questions.")


class DirectResponse(BaseModel):
    """A direct response to a question."""

    thought: str = Field(description="Short reasoning.")
    answer: str = Field(description="Direct final answer.")


class ContractQuestionResponse(BaseModel):
    """The result of contracting (simplifying) a question."""

    thought: str = Field(description="Reasoning on how the question was compressed.")
    question: str = Field(description="New, simplified, self-contained question.")


class EnsembleResponse(BaseModel):
    """The ensemble process result."""

    thought: str = Field(description="Explanation for choosing the final answer.")
    answer: str = Field(description="Best final answer after ensemble.")
    confidence: float = Field(description="Confidence score in [0, 1].")

    def model_post_init(self, __context) -> None:
        # Clamp confidence to [0, 1]
        self.confidence = max(0.0, min(1.0, self.confidence))


class LabelResponse(BaseModel):
    """A response used to refine sub-question dependencies and structure."""

    thought: str = Field(description="Explanation or reasoning about labeling.")
    sub_questions: list[SubQuestionNode] = Field(
        default_factory=list, description="Refined list of sub-questions with corrected dependencies."
    )
    # Some tasks also keep the final answer, but we focus on sub-questions.


# --------------- 1) Prompter Classes with multi-hop context ---------------


class BaseAoTPrompter(ABC):
    """
    Abstract base prompter that defines the required prompt methods.
    """

    @abstractmethod
    def recursive_decompose_prompt(
        self, question: str, sub_answers: Optional[str] = None, context: Optional[str] = None
    ) -> str: ...

    @abstractmethod
    def direct_prompt(self, question: str, context: Optional[str] = None) -> str: ...

    @abstractmethod
    def label_prompt(
        self, question: str, decompose_response: RecursiveDecomposeResponse, context: Optional[str] = None
    ) -> str:
        """
        Prompt used to 're-check' the sub-questions for correctness and dependencies.
        """

    @abstractmethod
    def contract_prompt(self, question: str, sub_answers: str, context: Optional[str] = None) -> str: ...

    @abstractmethod
    def ensemble_prompt(
        self,
        original_question: str,
        direct_answer: str,
        decompose_answer: str,
        contracted_direct_answer: str,
        context: Optional[str] = None,
    ) -> str: ...


class GeneralAoTPrompter(BaseAoTPrompter):
    """
    Generic prompter for non-specialized or 'general' queries.
    """

    def recursive_decompose_prompt(
        self, question: str, sub_answers: Optional[str] = None, context: Optional[str] = None
    ) -> str:
        sub_ans_str = f"\nSub-question answers:\n{sub_answers}" if sub_answers else ""
        context_str = f"\nCONTEXT:\n{context}" if context else ""
        return (
            "You are a highly analytical assistant skilled in breaking down complex problems.\n"
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

    def direct_prompt(self, question: str, context: Optional[str] = None) -> str:
        context_str = f"\nCONTEXT:\n{context}" if context else ""
        return (
            "You are a concise and insightful assistant.\n"
            "Provide a direct answer with a short reasoning.\n\n"
            "REQUIREMENTS:\n"
            "1. Return valid JSON:\n"
            "   {'thought': '...', 'answer': '...'}\n"
            "2. 'thought': Offer a brief reasoning.\n"
            "3. 'answer': Deliver a precise solution.\n\n"
            f"QUESTION:\n{question}{context_str}"
        )

    def label_prompt(
        self, question: str, decompose_response: RecursiveDecomposeResponse, context: Optional[str] = None
    ) -> str:
        context_str = f"\nCONTEXT:\n{context}" if context else ""
        return (
            "You have a set of sub-questions from a decomposition process.\n"
            "We want to correct or refine the dependencies between sub-questions.\n\n"
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
            "You are tasked with compressing or simplifying a complex question into a single self-contained one.\n\n"
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
        direct_answer: str,
        decompose_answer: str,
        contracted_direct_answer: str,
        context: Optional[str] = None,
    ) -> str:
        context_str = f"\nCONTEXT:\n{context}" if context else ""
        return (
            "You are an expert at synthesizing multiple candidate answers.\n"
            "Consider the following candidates:\n"
            f"1) Direct: {direct_answer}\n"
            f"2) Decomposition-based: {decompose_answer}\n"
            f"3) Contracted Direct: {contracted_direct_answer}\n\n"
            "REQUIREMENTS:\n"
            "1. Return valid JSON:\n"
            "   {'thought': '...', 'answer': '...', 'confidence': <float in [0,1]>}\n"
            "2. 'thought': Summarize how you decided.\n"
            "3. 'answer': Final best answer.\n"
            "4. 'confidence': Confidence score in [0,1].\n\n"
            f"ORIGINAL QUESTION:\n{original_question}"
            f"{context_str}"
        )


class MathAoTPrompter(GeneralAoTPrompter):
    """
    Specialized prompter for math questions; includes domain-specific hints.
    """

    def recursive_decompose_prompt(
        self, question: str, sub_answers: Optional[str] = None, context: Optional[str] = None
    ) -> str:
        base = super().recursive_decompose_prompt(question, sub_answers, context)
        return base + "\nFocus on mathematical rigor and step-by-step derivations.\n"

    def direct_prompt(self, question: str, context: Optional[str] = None) -> str:
        base = super().direct_prompt(question, context)
        return base + "\nEnsure mathematical correctness in your reasoning.\n"

    # label_prompt, contract_prompt, ensemble_prompt can also be overridden similarly if needed.


class CodingAoTPrompter(GeneralAoTPrompter):
    """
    Specialized prompter for coding/algorithmic queries.
    """

    def recursive_decompose_prompt(
        self, question: str, sub_answers: Optional[str] = None, context: Optional[str] = None
    ) -> str:
        base = super().recursive_decompose_prompt(question, sub_answers, context)
        return base + "\nBreak down into programming concepts or implementation steps.\n"


class PhilosophyAoTPrompter(GeneralAoTPrompter):
    """
    Specialized prompter for philosophical discussions.
    """

    def recursive_decompose_prompt(
        self, question: str, sub_answers: Optional[str] = None, context: Optional[str] = None
    ) -> str:
        base = super().recursive_decompose_prompt(question, sub_answers, context)
        return base + "\nConsider key philosophical theories and arguments.\n"


class MultiHopAoTPrompter(GeneralAoTPrompter):
    """
    Specialized prompter for multi-hop Q&A with explicit context usage.
    """

    def recursive_decompose_prompt(
        self, question: str, sub_answers: Optional[str] = None, context: Optional[str] = None
    ) -> str:
        base = super().recursive_decompose_prompt(question, sub_answers, context)
        return (
            base + "\nTreat this as a multi-hop question. Use the provided context carefully.\n"
            "Extract partial evidence from the context for each sub-question.\n"
        )

    def direct_prompt(self, question: str, context: Optional[str] = None) -> str:
        base = super().direct_prompt(question, context)
        return base + "\nFor multi-hop, ensure each piece of reasoning uses only the relevant parts of the context.\n"


# ----------------- 2) The AoTPipeline class with label + score + multi-hop -----------------

T = TypeVar("T", bound=BaseModel)


@dataclass
class AoTPipeline:
    """
    Implements an Atom-of-Thought pipeline with:
    1) Domain detection
    2) Direct solution
    3) Recursive Decomposition
    4) Label step to refine sub-questions
    5) Contract question
    6) Contracted direct solution
    7) Ensemble
    8) (Optional) Score final answer if ground-truth is available
    """

    chatterer: Chatterer
    max_depth: int = 2
    max_retries: int = 2

    prompter_map: dict[Domain, BaseAoTPrompter] = field(
        default_factory=lambda: {
            Domain.GENERAL: GeneralAoTPrompter(),
            Domain.MATH: MathAoTPrompter(),
            Domain.CODING: CodingAoTPrompter(),
            Domain.PHILOSOPHY: PhilosophyAoTPrompter(),
            Domain.MULTIHOP: MultiHopAoTPrompter(),
        }
    )

    async def _ainvoke_pydantic(self, prompt: str, model_cls: Type[T], default_answer: str = "Unable to process") -> T:
        """
        Attempts up to max_retries to parse the model_cls from LLM output.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                result = await self.chatterer.agenerate_pydantic(
                    response_model=model_cls, messages=[{"role": "user", "content": prompt}]
                )
                logger.debug(f"[Validated attempt {attempt}] => {result.model_dump_json(indent=2)}")
                return result
            except ValidationError as e:
                logger.warning(f"Validation error on attempt {attempt}/{self.max_retries}: {str(e)}")
                if attempt == self.max_retries:
                    # Return an empty or fallback
                    if model_cls == EnsembleResponse:
                        return model_cls(thought="Failed label parse", answer=default_answer, confidence=0.0)
                    elif model_cls == ContractQuestionResponse:
                        return model_cls(thought="Failed contract parse", question="Unknown")
                    elif model_cls == LabelResponse:
                        return model_cls(thought="Failed label parse", sub_questions=[])
                    else:
                        # fallback for Direct/Decompose
                        return model_cls(thought="Failed parse", answer=default_answer)
        raise RuntimeError("Unexpected error in _ainvoke_pydantic")

    async def _adetect_domain(self, question: str, context: Optional[str] = None) -> Domain:
        """
        Queries an LLM to figure out which domain is best suited.
        """

        class InferredDomain(BaseModel):
            domain: Domain

        ctx_str = f"\nCONTEXT:\n{context}" if context else ""
        domain_prompt = (
            "You are an expert domain classifier. "
            "Possible domains: [general, math, coding, philosophy, multihop].\n\n"
            "Return valid JSON: {'domain': '...'}.\n"
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
        self, question: str, depth: int, prompter: BaseAoTPrompter, context: Optional[str] = None
    ) -> RecursiveDecomposeResponse:
        """
        Recursively decomposes a question into sub-questions, applying an optional label step.
        """
        if depth < 0:
            return RecursiveDecomposeResponse(thought="Max depth reached", final_answer="Unknown", sub_questions=[])

        indent: LiteralString = "  " * (self.max_depth - depth)
        logger.debug(f"{indent}Decomposing at depth {self.max_depth - depth}: {question}")

        # Step 1: Base decomposition
        prompt = prompter.recursive_decompose_prompt(question, context=context)
        decompose_resp: RecursiveDecomposeResponse = await self._ainvoke_pydantic(prompt, RecursiveDecomposeResponse)

        # Step 2: Label step to refine sub-questions if any
        if decompose_resp.sub_questions:
            label_prompt: str = prompter.label_prompt(question, decompose_resp, context=context)
            label_resp: LabelResponse = await self._ainvoke_pydantic(label_prompt, LabelResponse)
            # Overwrite the sub-questions with refined ones
            decompose_resp.sub_questions = label_resp.sub_questions

        # Step 3: If depth > 0, try to resolve sub-questions recursively
        #         so we can potentially update final_answer
        if depth > 0 and decompose_resp.sub_questions:
            resolved_subs: list[SubQuestionNode] = await self._aresolve_sub_questions(
                decompose_resp.sub_questions, depth, prompter, context
            )
            # Step 4: Re-invoke decomposition with known sub-answers (sub_answers)
            sub_answers_str: str = "\n".join(f"{sq.question}: {sq.answer}" for sq in resolved_subs if sq.answer)
            if sub_answers_str:
                refine_prompt: str = prompter.recursive_decompose_prompt(question, sub_answers_str, context=context)
                refined_resp: RecursiveDecomposeResponse = await self._ainvoke_pydantic(
                    refine_prompt, RecursiveDecomposeResponse
                )
                # Use the refined final answer, keep resolved sub-questions
                decompose_resp.final_answer = refined_resp.final_answer
                decompose_resp.sub_questions = resolved_subs

        return decompose_resp

    async def _aresolve_sub_questions(
        self, sub_questions: list[SubQuestionNode], depth: int, prompter: BaseAoTPrompter, context: Optional[str] = None
    ) -> list[SubQuestionNode]:
        """
        Resolves each sub-question (potentially reusing the same decomposition approach)
        in a topological order of dependencies.
        """
        n: int = len(sub_questions)
        resolved: dict[int, SubQuestionNode] = {}

        # Build adjacency
        in_degree: list[int] = [0] * n
        graph: list[list[int]] = [[] for _ in range(n)]
        for i, sq in enumerate(sub_questions):
            for dep in sq.depend:
                if 0 <= dep < n:
                    in_degree[i] += 1
                    graph[dep].append(i)

        # Topological BFS
        queue: list[int] = [i for i in range(n) if in_degree[i] == 0]
        order: list[int] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for nxt in graph[node]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        # If there's a cycle, some sub-questions won't appear in order
        # but we'll attempt to resolve what we can
        async def resolve_single_subq(idx: int) -> None:
            sq: SubQuestionNode = sub_questions[idx]
            # Attempt to answer this sub-question by decomposition if needed
            sub_decomp: RecursiveDecomposeResponse = await self._arecursive_decompose_question(
                sq.question, depth - 1, prompter, context
            )
            sq.answer = sub_decomp.final_answer
            resolved[idx] = sq

        await asyncio.gather(*(resolve_single_subq(i) for i in order))

        # Return only resolved sub-questions
        return [resolved[i] for i in range(n) if i in resolved]

    def _calculate_score(self, answer: str, ground_truth: Optional[str], domain: Domain) -> float:
        """
        Example scoring function. Real usage depends on having ground-truth.
        If ground_truth is None, returns -1.0 as 'no score possible'.

        This function is used for `post-hoc` scoring of the final answer.
        """
        if ground_truth is None:
            return -1.0

        # Very simplistic example:
        # MATH: attempt numeric equality
        if domain == Domain.MATH:
            try:
                ans_val = float(answer.strip())
                gt_val = float(ground_truth.strip())
                return 1.0 if abs(ans_val - gt_val) < 1e-9 else 0.0
            except ValueError:
                return 0.0

        # For anything else, do a naive exact-match check ignoring case
        return 1.0 if answer.strip().lower() == ground_truth.strip().lower() else 0.0

    async def arun_pipeline(
        self, question: str, context: Optional[str] = None, ground_truth: Optional[str] = None
    ) -> str:
        """
        Full AoT pipeline. If ground_truth is provided, we compute a final score.
        """
        # 1) Domain detection
        domain = await self._adetect_domain(question, context)
        prompter = self.prompter_map.get(domain, GeneralAoTPrompter())
        logger.debug(f"Detected domain: {domain}")

        # 2) Direct approach
        direct_prompt = prompter.direct_prompt(question, context)
        direct_resp = await self._ainvoke_pydantic(direct_prompt, DirectResponse)
        direct_answer = direct_resp.answer
        logger.debug(f"Direct answer => {direct_answer}")

        # 3) Recursive Decomposition + label
        decomp_resp = await self._arecursive_decompose_question(question, self.max_depth, prompter, context)
        decompose_answer = decomp_resp.final_answer
        logger.debug(f"Decomposition answer => {decompose_answer}")

        # 4) Contract question
        sub_answers_str = "\n".join(f"{sq.question}: {sq.answer}" for sq in decomp_resp.sub_questions if sq.answer)
        contract_prompt = prompter.contract_prompt(question, sub_answers_str, context)
        contract_resp = await self._ainvoke_pydantic(contract_prompt, ContractQuestionResponse)
        contracted_question = contract_resp.question
        logger.debug(f"Contracted question => {contracted_question}")

        # 5) Direct approach on contracted question
        contracted_direct_prompt = prompter.direct_prompt(contracted_question, context)
        contracted_direct_resp = await self._ainvoke_pydantic(contracted_direct_prompt, DirectResponse)
        contracted_direct_answer = contracted_direct_resp.answer
        logger.debug(f"Contracted direct answer => {contracted_direct_answer}")

        # 6) Ensemble
        ensemble_prompt = prompter.ensemble_prompt(
            original_question=question,
            direct_answer=direct_answer,
            decompose_answer=decompose_answer,
            contracted_direct_answer=contracted_direct_answer,
            context=context,
        )
        ensemble_resp = await self._ainvoke_pydantic(ensemble_prompt, EnsembleResponse)
        final_answer = ensemble_resp.answer
        logger.debug(f"Ensemble final answer => {final_answer} (confidence={ensemble_resp.confidence})")

        # 7) Optional scoring
        final_score = self._calculate_score(final_answer, ground_truth, domain)
        if final_score >= 0.0:
            logger.info(f"Final Score: {final_score:.3f} (domain={domain})")

        return final_answer


# ------------------ 3) AoTStrategy that uses the pipeline ------------------


@dataclass(kw_only=True)
class AoTStrategy(BaseStrategy):
    pipeline: AoTPipeline

    async def ainvoke(self, messages: LanguageModelInput) -> str:
        logger.debug(f"Invoking with messages: {messages}")

        input = self.pipeline.chatterer.client._convert_input(messages)
        input_string = input.to_string()
        logger.debug(f"Extracted question: {input_string}")
        return await self.pipeline.arun_pipeline(input_string)

    def invoke(self, messages: LanguageModelInput) -> str:
        return asyncio.run(self.ainvoke(messages))


# ------------------ 4) Example usage (main) ------------------
if __name__ == "__main__":
    from warnings import filterwarnings

    import colorama

    filterwarnings("ignore", category=UserWarning)

    colorama.init(autoreset=True)

    class ColoredFormatter(logging.Formatter):
        COLORS = {
            "DEBUG": colorama.Fore.CYAN,
            "INFO": colorama.Fore.GREEN,
            "WARNING": colorama.Fore.YELLOW,
            "ERROR": colorama.Fore.RED,
            "CRITICAL": colorama.Fore.RED + colorama.Style.BRIGHT,
        }

        def format(self, record):
            levelname = record.levelname
            message = super().format(record)
            return f"{self.COLORS.get(levelname, colorama.Fore.WHITE)}{message}{colorama.Style.RESET_ALL}"

    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    file_handler = logging.FileHandler("atom_of_thoughts.log", encoding="utf-8", mode="w")
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.propagate = False

    pipeline = AoTPipeline(chatterer=Chatterer.openai(), max_depth=2)
    strategy = AoTStrategy(pipeline=pipeline)

    question = "What would Newton discover if hit by an apple falling from 100 meters?"
    answer = strategy.invoke(question)
    print(answer)
