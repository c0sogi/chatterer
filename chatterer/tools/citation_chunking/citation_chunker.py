import logging
from typing import Callable, NamedTuple, Optional, Self

import colorama
from colorama import Fore

from ...language_model import Chatterer
from .chunks import CitationChunks
from .citations import Citations
from .prompt import (
    generate_fewshot_affirmative_response,
    generate_human_assistant_fewshot_examples,
    generate_instruction,
)

logger = logging.getLogger(__name__)
colorama.init()


class GlobalCoverage(NamedTuple):
    coverage: float
    matched_intervals: list[tuple[int, int]]

    @staticmethod
    def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not intervals:
            return []
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged: list[tuple[int, int]] = [sorted_intervals[0]]
        for current in sorted_intervals[1:]:
            prev = merged[-1]
            if current[0] <= prev[1]:
                merged[-1] = (prev[0], max(prev[1], current[1]))
            else:
                merged.append(current)
        return merged

    @classmethod
    def from_verified_citations(cls, verified_chunks: list[Citations], document: str) -> Self:
        all_intervals: list[tuple[int, int]] = []
        for chunk in verified_chunks:
            for matches in chunk.references.values():
                for m in matches:
                    all_intervals.append((m.start_idx, m.end_idx))
        merged: list[tuple[int, int]] = cls.merge_intervals(all_intervals)
        doc_length: int = len(document)
        total_matched = sum((e - s for s, e in merged))
        coverage: float = total_matched / doc_length if doc_length > 0 else 0.0
        return cls(coverage=coverage, matched_intervals=merged)


def citation_chunker(
    document: str,
    chatterer: Chatterer,
    global_coverage_threshold: float = 0.9,
    num_refinement_steps: int = 3,
    fewshot_examples_generator: Optional[
        Callable[[], list[tuple[str, str]]]
    ] = generate_human_assistant_fewshot_examples,
    instruction_generator: Optional[Callable[[], str]] = generate_instruction,
    fewshot_affirmative_response: Optional[Callable[[], str]] = generate_fewshot_affirmative_response,
    test_global_coverage: bool = False,
) -> list[Citations]:
    """
    1) Obtain CitationChunks via the LLM.
    2) Process each chunk to extract MatchedText using snippet-based index correction.
    3) Calculate overall document coverage and print results.
    """
    unverified_chunks: CitationChunks = CitationChunks.from_llm(
        chatterer=chatterer,
        document=document,
        fewshot_examples_generator=fewshot_examples_generator,
        instruction_generator=instruction_generator,
        fewshot_affirmative_response=fewshot_affirmative_response,
    )

    verified_chunks: list[Citations] = []
    for chunk in unverified_chunks.citation_chunks:
        try:
            vc: Citations = Citations.from_unverified(
                unverified_chunk=chunk,
                document=document,
                model_and_refinement_steps=(chatterer, num_refinement_steps),
            )
            verified_chunks.append(vc)
        except Exception as e:
            logger.error(f"Error processing chunk for subject '{chunk.subject}': {e}")

    if test_global_coverage:
        gc = GlobalCoverage.from_verified_citations(verified_chunks, document)
        logger.info(f"Global coverage: {gc.coverage * 100:.1f}%")
        if gc.coverage < global_coverage_threshold:
            logger.info(
                f"Global coverage {gc.coverage * 100:.1f}% is below the threshold {global_coverage_threshold * 100:.1f}%."
            )
        print("=== Final Global Coverage Check ===")
        print(f"Overall coverage: {gc.coverage * 100:.1f}% of the document.")
        if gc.matched_intervals:
            print("Merged matched intervals:")
            for interval in gc.matched_intervals:
                print(f" - {interval}")
        else:
            print("No matches found across all chunks.")
        print("\n=== Raw Semantic Chunking Result ===")
        for vc in verified_chunks:
            print(f"{Fore.LIGHTGREEN_EX}[SUBJECT] {Fore.GREEN}{vc.name}{Fore.RESET}")
            if vc.references:
                for source_key, matches in vc.references.items():
                    print(f"{Fore.LIGHTBLUE_EX}  [SOURCE] {Fore.BLUE}{source_key}{Fore.RESET}")
                    for mt in matches:
                        snippet = repr(mt.text)
                        print(
                            f"    {Fore.LIGHTYELLOW_EX}[MATCH @ {mt.start_idx}~{mt.end_idx}] {Fore.YELLOW}{snippet}{Fore.RESET}"
                        )
            else:
                print(" - (No matches found even after refinement.)")

    return verified_chunks
