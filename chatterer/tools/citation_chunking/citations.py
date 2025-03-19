from __future__ import annotations

import difflib
import logging
from typing import NamedTuple, Optional, Self, TypeAlias

from pydantic import Field
from regex import DOTALL
from regex import compile as regex_compile
from regex import error as regex_error

from ...language_model import Chatterer
from ...messages import HumanMessage
from .chunks import CitationChunk
from .reference import MultiMatchRegex, Reference, SingleMatchCitation
from .utils import MatchedText

ModelAndSteps: TypeAlias = tuple[Chatterer, int]
logger = logging.getLogger(__name__)


class Citations(NamedTuple):
    """
    Holds the verified citation chunks and their matching information.
    """

    name: str
    references: dict[Reference, list[ReferencedTextMatch]]

    @classmethod
    def from_unverified(
        cls,
        unverified_chunk: CitationChunk,
        document: str,
        model_and_refinement_steps: Optional[ModelAndSteps] = None,  # Optional LLM for refinement
    ) -> Self:
        subject: str = unverified_chunk.subject
        self: Self = cls(name=subject, references={})
        for reference in unverified_chunk.references or ():
            if isinstance(reference, SingleMatchCitation):
                try:
                    mt: Optional[ReferencedTextMatch] = ReferencedTextMatch.from_citation(
                        subject=subject,
                        citation=reference,
                        document=document,
                        model_and_refinement_steps=model_and_refinement_steps,
                    )
                    if mt is None or not mt.text.strip():
                        logger.warning(f"Failed to extract text for citation {reference} in subject '{subject}'.")
                    else:
                        self.references[reference] = [mt]
                except Exception as e:
                    logger.error(f"Error processing citation {reference} for subject '{subject}': {e}")
            else:
                try:
                    regex_matches: list[ReferencedTextMatch] = ReferencedTextMatch.from_regex(
                        regex=reference, subject=subject, document=document
                    )
                    if regex_matches:
                        self.references[reference] = regex_matches
                except regex_error as e:
                    logger.error(f"Regex error for subject '{subject}' with pattern '{reference}': {e}")
        return self


class ReferencedTextMatch(MatchedText):
    @classmethod
    def from_citation(
        cls,
        subject: str,
        citation: SingleMatchCitation,
        document: str,
        model_and_refinement_steps: Optional[ModelAndSteps] = None,  # Optional LLM for quality-check refinement
    ) -> Optional[Self]:
        """
        Extract text from the document using the adjusted citation indices.
        Additionally, if a language model is provided, evaluate the extraction quality
        and refine it if needed.
        """
        citation_id: Optional[SingleMatchCitationWithIndex] = SingleMatchCitationWithIndex.from_indexless_citation(
            indexless_citation=citation,
            document=document,
            subject=subject,
            model_and_refinement_steps=model_and_refinement_steps,
        )
        if citation_id is None:
            return

        return cls(
            start_idx=citation_id.start,
            end_idx=citation_id.end,
            text=citation_id.extracted_text,
        )

    @classmethod
    def from_regex(cls, regex: MultiMatchRegex, subject: str, document: str) -> list[Self]:
        """
        Apply the given regex to the document and return all matching results as a list of MatchedText.
        """
        try:
            compiled_pattern = regex_compile(regex.regular_expression, flags=DOTALL)
        except regex_error as e:
            logger.error(f"Regex compilation error for pattern /{regex.regular_expression}/: {e}")
            raise e
        try:
            matches = list(compiled_pattern.finditer(document, timeout=1.0))
        except regex_error as e:
            logger.error(f"Regex matching error for pattern /{regex.regular_expression}/: {e}")
            raise e
        return [cls(start_idx=m.start(), end_idx=m.end(), text=m.group()) for m in matches]


class SingleMatchCitationWithIndex(SingleMatchCitation):
    start: int = Field(description="The computed start index of the citation in the document.")
    end: int = Field(description="The computed end index of the citation in the document.")
    extracted_text: str = Field(description="The extracted text from the document using the computed indices.")

    @classmethod
    def from_indexless_citation(
        cls,
        indexless_citation: SingleMatchCitation,
        document: str,
        subject: str,
        model_and_refinement_steps: Optional[ModelAndSteps] = None,  # Optional LLM for quality-check refinement
    ) -> Optional[Self]:
        """
        Compute the correct start and end indices for the citation based on the provided text snippets.
        This method ignores any indices provided by the LLM and computes them using a similarity-based search.
        If multiple high-scoring candidates are found, the one with the highest effective score is chosen.
        """
        if model_and_refinement_steps is None:
            model = None
            num_refinement_steps = 1
        else:
            model, num_refinement_steps = model_and_refinement_steps
        for _ in range(num_refinement_steps):
            result = cls.from_indexless_citation_with_refinement(
                indexless_citation=indexless_citation,
                document=document,
                subject=subject,
                chatterer=model,
            )
            if result is None:
                continue
            return result

    @staticmethod
    def find_best_match_index(snippet: str, document: str, target_index: int) -> Optional[int]:
        """
        Extracts a candidate window centered around the specified target_index,
        with a size equal to the length of the snippet. Within this region,
        it calculates the similarity with the snippet using a sliding window approach.

        The index of the candidate with the highest effective_score is returned.
        If no suitable candidate is found, the target_index is returned.

        Note: If multiple high-scoring candidates are found, the one with the highest effective score is chosen.
        """
        snippet = snippet.strip()
        if not snippet:
            return
        snippet_len: int = len(snippet)
        best_index: int = -1
        best_effective_score = 0.0
        max_radius = max(target_index, len(document) - target_index)
        for offset in range(max_radius):
            for candidate_index in (
                target_index - offset,
                target_index + offset,
            ):
                if candidate_index < 0 or candidate_index + snippet_len > len(document):
                    continue
                candidate_segment = document[candidate_index : min(candidate_index + snippet_len, len(document))]
                if len(candidate_segment) < snippet_len:
                    continue
                local_best_similarity = 0.0
                local_best_offset = 0
                for i in range(0, len(candidate_segment) - snippet_len + 1):
                    candidate_window = candidate_segment[i : i + snippet_len]
                    similarity = difflib.SequenceMatcher(None, snippet, candidate_window).ratio()
                    if similarity > local_best_similarity:
                        local_best_similarity = similarity
                        local_best_offset = i
                candidate_final_index = candidate_index + local_best_offset
                if candidate_final_index + snippet_len > len(document):
                    candidate_final_index = len(document) - snippet_len
                if local_best_similarity > best_effective_score:
                    best_effective_score = local_best_similarity
                    best_index = candidate_final_index
        if not 0 <= best_index < len(document):
            logger.warning(f"Snippet '{snippet}' not found with sufficient similarity.")
            return
        else:
            logger.debug(
                f"Found best match for snippet '{snippet}' at index {best_index} with effective score {best_effective_score:.2f}."
            )
            return best_index

    @classmethod
    def from_indexless_citation_with_refinement(
        cls,
        indexless_citation: SingleMatchCitation,
        document: str,
        subject: str,
        chatterer: Optional[Chatterer],
    ) -> Optional[Self]:
        if chatterer is None:
            logger.error("No LLM provided for indexless citation refinement.")
            new_indexless_citation = indexless_citation
        else:
            new_indexless_citation = chatterer.generate_pydantic(
                response_model=SingleMatchCitation,
                messages=[
                    HumanMessage(
                        content=(
                            "I tried to find the `SNIPPET` in the `original-raw-document` to extract a text citation for the subject `subject-to-parse`, but I couldn't find it. "
                            "Please provide `citation-start-from` and `citation-end-at` to help me locate the correct text span.\n"
                            "---\n"
                            "<original-raw-document>\n"
                            f"{document}\n"
                            "</original-raw-document>\n"
                            "---\n"
                            "<subject-to-parse>\n"
                            f"{subject}\n"
                            "</subject-to-parse>\n"
                            "---\n"
                            "<current-citation-start-from>\n"
                            f"{indexless_citation.start_from}\n"
                            "</current-citation-start-from>\n"
                            "---\n"
                            "<current-citation-end-at>\n"
                            f"{indexless_citation.end_at}\n"
                            "</current-citation-end-at>\n"
                        )
                    ),
                ],
            )
        doc_len: int = len(document)

        start_snippet: str = new_indexless_citation.start_from.strip()
        if start_snippet:
            target_for_start = document.find(start_snippet)
            if target_for_start == -1:
                target_for_start = 0
            new_start: Optional[int] = cls.find_best_match_index(
                snippet=start_snippet,
                document=document,
                target_index=target_for_start,
            )
            if new_start is None:
                return
        else:
            logger.warning("No start_text provided")
            return
        end_snippet: str = new_indexless_citation.end_at.strip()
        if end_snippet:
            target_for_end = document.find(end_snippet, new_start)
            if target_for_end == -1:
                target_for_end = new_start
            candidate_end: Optional[int] = cls.find_best_match_index(
                snippet=end_snippet,
                document=document,
                target_index=target_for_end,
            )
            if candidate_end is None:
                return
            new_end: int = candidate_end + len(end_snippet)
        else:
            logger.warning("No end_text provided; defaulting end index to document length.")
            new_end = doc_len
        if not 0 <= new_start < new_end <= doc_len:
            logger.error(f"Adjusted citation indices invalid: start {new_start}, end {new_end}, doc_len {doc_len}.")
            return
        try:
            extracted_text = document[new_start:new_end]
        except IndexError as e:
            logger.error(f"Error extracting text using adjusted citation indices: {e}")
            return
        return cls(
            start=new_start,
            end=new_end,
            start_from=new_indexless_citation.start_from,
            end_at=new_indexless_citation.end_at,
            extracted_text=extracted_text,
        )
