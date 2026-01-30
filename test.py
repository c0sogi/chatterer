"""Example usage of AoT pipeline with rich results and progress callbacks."""

import asyncio

from chatterer import Chatterer
from chatterer.strategies import AoTResult, RichProgressCallback, aot_graph, aot_input, aot_output


async def main():
    """Demonstrate AoT pipeline with progress tracking."""

    # Rich real-time progress visualization
    with RichProgressCallback() as progress:
        graph = aot_graph(
            chatterer=Chatterer.from_provider(
                "openai:gpt-5-nano",
                option={"kwargs": {"reasoning_effort": "minimal"}},
            ),
            max_depth=3,
            max_sub_questions=3,
            on_progress=progress,
        )
        result: AoTResult = aot_output(
            await graph.ainvoke(  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
                aot_input(
                    messages="What are the key benefits of using LangGraph for AI pipelines?",
                    timeout=30,
                )
            )
        )

    # Pretty print the results
    result.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
