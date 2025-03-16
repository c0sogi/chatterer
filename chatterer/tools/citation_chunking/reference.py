from typing import Literal, TypeAlias

from pydantic import BaseModel, Field


class MultiMatchRegex(BaseModel):
    type: Literal["multi_match_regex"] = Field(
        description="A regex pattern that should match multiple instances of the subject in the document."
    )
    regular_expression: str = Field(
        description="The regex pattern that should match multiple instances of the subject in the document."
    )

    def __hash__(self) -> int:
        return hash((self.type, self.regular_expression))


class SingleMatchCitation(BaseModel):
    start_from: str = Field(description="A snippet of text at the beginning of the cited section.")
    end_at: str = Field(description="A snippet of text at the end of the cited section.")

    def __hash__(self) -> int:
        return hash((self.start_from, self.end_at))


Reference: TypeAlias = SingleMatchCitation | MultiMatchRegex
