"""
ragent/prompt/citation_chunking.py

This module defines prompt constants for citation chunking.
The LLM is expected to return JSON objects that include only the text snippets for the beginning and end of the citation span.
The character indices will be computed in a post‐processing step.
"""

from functools import cache


@cache
def generate_instruction() -> str:
    from .chunks import CitationChunk, CitationChunks
    from .reference import (
        MultiMatchRegex,
        SingleMatchCitation,
    )

    return (
        "You are an AI specialized in 'citation-based text chunking'.\n"
        "Given a document, perform the following steps:\n"
        "1) Identify the major topics in the document.\n"
        "2) For each topic, provide a list of citation objects indicating the text snippets at the beginning and end of the relevant paragraph(s) for that topic.\n\n"
        "Important:\n"
        "- Return citation objects with 'start_text' and 'end_text' fields to precisely capture the text span. Do NOT include character indices.\n"
        "- If a regular expression based matching is more appropriate for a topic (e.g. for multiple matches), you may include a regex object of type 'multi_match_regex'.\n\n"
        "Return JSON strictly in the following format:\n"
        "{json_example}\n\n"
        "1) Return only valid JSON (no extra keys).\n"
        "2) Do NOT include any commentary.\n"
        "3) Ensure that the citations capture the entire relevant paragraph without overlap or omission."
    ).format(
        json_example=CitationChunks(
            citation_chunks=[
                CitationChunk(
                    subject="Quantum Advantage",
                    references=[
                        SingleMatchCitation(
                            start_from="Starting snippet...",
                            end_at="... Ending snippet",
                        ),
                        MultiMatchRegex(
                            type="multi_match_regex",
                            regular_expression="Some.*?regex.*?pattern",
                        ),
                    ],
                ),
            ]
        ).model_dump_json(indent=2)
    )


@cache
def generate_human_assistant_fewshot_examples() -> list[tuple[str, str]]:
    from .chunks import CitationChunk, CitationChunks
    from .reference import SingleMatchCitation

    return [
        (
            "Agent-Semantic Chunking of the following text:\n\n"
            "Title: Revolutionary Breakthrough in Quantum Computing\n\n"
            "In a landmark development, researchers at the National Quantum Laboratory unveiled a quantum computer "
            "that demonstrates clear quantum advantage by performing computations that are infeasible on classical systems.\n\n"
            "The breakthrough is the result of years of rigorous research and international collaboration. "
            "The system leverages entanglement and superposition to process complex algorithms at unprecedented speeds.\n\n"
            "However, practical applications are still emerging, and experts caution about scalability challenges. "
            "Meanwhile, several tech giants are expressing keen interest in integrating quantum technology into future products.\n\n"
            "Please classify the major topics and return the exact text snippets (for the start and end of the relevant paragraphs) for each topic.",
            CitationChunks(
                citation_chunks=[
                    CitationChunk(
                        subject="Quantum Advantage",
                        references=[
                            SingleMatchCitation(
                                start_from="In a landmark development",
                                end_at="on classical systems.",
                            ),
                        ],
                    ),
                    CitationChunk(
                        subject="Research Collaboration",
                        references=[
                            SingleMatchCitation(
                                start_from="The breakthrough is the result",
                                end_at="unprecedented speeds.",
                            ),
                        ],
                    ),
                    CitationChunk(
                        subject="Practical Challenges",
                        references=[
                            SingleMatchCitation(
                                start_from="However, practical applications",
                                end_at="scalability challenges.",
                            ),
                        ],
                    ),
                    CitationChunk(
                        subject="Industry Interest",
                        references=[
                            SingleMatchCitation(
                                start_from="Meanwhile, several tech giants",
                                end_at="future products.",
                            ),
                        ],
                    ),
                ]
            ).model_dump_json(indent=2),
        ),
        (
            "Agent-Semantic Chunking of the following text:\n\n"
            "Title: Rising Seas and Coastal Erosion: A Global Crisis\n\n"
            "Communities worldwide face the impacts of climate change as rising sea levels lead to accelerated coastal erosion, "
            "jeopardizing homes and critical infrastructure.\n\n"
            'In a small coastal town, residents noted that "the encroaching sea" has already begun to claim beachfront properties, '
            "prompting local authorities to implement emergency measures.\n\n"
            "Environmental experts warn that without significant intervention, the frequency and severity of these events will increase, "
            "further exacerbating the global climate crisis.\n\n"
            "Please classify the major topics and return the exact text snippets (for the start and end of the relevant paragraphs) for each topic.",
            CitationChunks(
                citation_chunks=[
                    CitationChunk(
                        subject="Coastal Erosion Impact",
                        references=[
                            SingleMatchCitation(
                                start_from="Communities worldwide face the impacts",
                                end_at="critical infrastructure.",
                            ),
                        ],
                    ),
                    CitationChunk(
                        subject="Local Emergency Response",
                        references=[
                            SingleMatchCitation(
                                start_from="In a small coastal town",
                                end_at="emergency measures.",
                            ),
                        ],
                    ),
                    CitationChunk(
                        subject="Expert Warning",
                        references=[
                            SingleMatchCitation(
                                start_from="Environmental experts warn",
                                end_at="global climate crisis.",
                            ),
                        ],
                    ),
                ]
            ).model_dump_json(indent=2),
        ),
    ]


def generate_fewshot_affirmative_response() -> str:
    return "Great! I will now perform the citation-based chunking. Please provide the document to process!"
