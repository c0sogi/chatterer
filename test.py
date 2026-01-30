"""Example usage of AoT pipeline with rich results and progress callbacks."""

import asyncio

from chatterer import Chatterer
from chatterer.strategies import RichProgressCallback, aot_graph, aot_input, aot_output
from chatterer.strategies.atom_of_thoughts import AoTResult


async def main():
    """Demonstrate AoT pipeline with progress tracking."""

    # Rich real-time progress visualization
    with RichProgressCallback() as progress:
        chatterer = Chatterer.openrouter("z-ai/glm-4.7-flash", option={"kwargs": {"reasoning_effort": "minimal"}})
        graph = aot_graph(chatterer, max_depth=3, max_sub_questions=3, on_progress=progress)
        result: AoTResult = aot_output(
            await graph.ainvoke(  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
                aot_input(
                    "What are the key benefits of using LangGraph for AI pipelines?",
                    timeout=60,
                )
            )
        )

    # Access rich results
    print("\n" + "=" * 80)
    print("FINAL ANSWER:")
    print(result.answer)
    print(f"\nConfidence: {result.confidence:.2f}")

    print("\n" + "=" * 80)
    print("INTERMEDIATE RESULTS:")
    if result.direct_answer:
        print(f"\nDirect Answer: {result.direct_answer}...")
    if result.decompose_answer:
        print(f"\nDecomposed Answer: {result.decompose_answer}...")

    if result.sub_questions:
        print(f"\nSub-questions ({len(result.sub_questions)}):")
        for i, sq in enumerate(result.sub_questions, 1):
            print(f"  {i}. Q: {sq.question}")
            print(f"     A: {sq.answer}...")

    if result.contracted_question:
        print(f"\nContracted Question: {result.contracted_question}")


if __name__ == "__main__":
    asyncio.run(main())
