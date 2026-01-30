"""Example usage of AoT pipeline with rich results and progress callbacks."""

from chatterer import Chatterer
from chatterer.strategies import RichProgressCallback, aot_invoke


def main():
    """Demonstrate AoT pipeline with progress tracking."""

    # Rich real-time progress visualization
    with RichProgressCallback() as progress:
        result = aot_invoke(
            Chatterer.openai("gpt-5-nano"),
            question="What are the key benefits of using LangGraph for AI pipelines?",
            max_depth=3,
            max_sub_questions=3,
            max_workers=4,
            on_progress=progress,
        )

    # Access rich results
    print("\n" + "=" * 80)
    if "answer" in result:
        print("FINAL ANSWER:")
        print(result["answer"])  # or str(result)
    if "confidence" in result:
        print(f"\nConfidence: {result['confidence']:.2f}")

    print("\n" + "=" * 80)
    print("INTERMEDIATE RESULTS:")
    if "direct_answer" in result:
        print(f"\nDirect Answer: {result['direct_answer']}...")
    if "decompose_answer" in result:
        print(f"\nDecomposed Answer: {result['decompose_answer']}...")

    if "sub_questions" in result and result["sub_questions"]:
        print(f"\nSub-questions ({len(result['sub_questions'])}):")
        for i, sq in enumerate(result["sub_questions"], 1):
            print(f"  {i}. Q: {sq.question}")
            print(f"     A: {sq.answer}...")

    if "contracted_question" in result:
        print(f"\nContracted Question: {result['contracted_question']}")


if __name__ == "__main__":
    main()
