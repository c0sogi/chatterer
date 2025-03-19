import logging
from typing import Callable, Optional, Self

from pydantic import BaseModel, Field

from ...language_model import Chatterer
from ...messages import AIMessage, BaseMessage, HumanMessage
from .prompt import (
    generate_fewshot_affirmative_response,
    generate_human_assistant_fewshot_examples,
    generate_instruction,
)
from .reference import Reference

logger = logging.getLogger(__name__)


class CitationChunk(BaseModel):
    subject: str = Field(description="The main topic or subject that the citations capture.")
    references: list[Reference] = Field(description="A list of citation objects and/or regex patterns for the subject.")


class CitationChunks(BaseModel):
    citation_chunks: list[CitationChunk] = Field(
        description="A list of citation chunks, each capturing a specific topic in the document."
    )

    @classmethod
    def from_llm(
        cls,
        chatterer: Chatterer,
        document: str,
        fewshot_examples_generator: Optional[
            Callable[[], list[tuple[str, str]]]
        ] = generate_human_assistant_fewshot_examples,
        instruction_generator: Optional[Callable[[], str]] = generate_instruction,
        fewshot_affirmative_response: Optional[Callable[[], str]] = generate_fewshot_affirmative_response,
    ) -> Self:
        messages: list[BaseMessage] = []
        if instruction_generator:
            messages.append(HumanMessage(content=instruction_generator()))
        if fewshot_examples_generator is not None:
            if fewshot_affirmative_response:
                messages.append(AIMessage(content=generate_fewshot_affirmative_response()))
            for human_ask, ai_answer in fewshot_examples_generator():
                messages.append(HumanMessage(content=human_ask))
                messages.append(AIMessage(content=ai_answer))
        messages.append(HumanMessage(content=document))
        try:
            return chatterer.generate_pydantic(response_model=cls, messages=messages)
        except Exception as e:
            logger.error(f"Error obtaining CitationChunks from LLM: {e}")
            raise e
