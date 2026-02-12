"""
Atom-of-Thought (AoT) Reasoning CLI

A command-line tool for multi-path reasoning using the AoT pipeline.
Decomposes questions, answers them in parallel, and synthesizes a final answer.
"""

import asyncio
import sys
from typing import Optional, cast

import typer

from chatterer import Chatterer
from chatterer.constants import DEFAULT_GOOGLE_MODEL
from chatterer.strategies.atom_of_thoughts import AoTState


def command(
    question: str = typer.Argument(help="The question to reason about."),
    chatterer: str = typer.Option(f"google:{DEFAULT_GOOGLE_MODEL}", help=f"Chatterer instance configuration (e.g., 'google:{DEFAULT_GOOGLE_MODEL}')."),
    max_depth: int = typer.Option(2, help="Maximum recursion depth for decomposition."),
    max_sub_questions: int = typer.Option(3, help="Maximum sub-questions per decomposition level."),
    max_workers: int = typer.Option(4, help="Maximum parallel workers for sub-question processing."),
    timeout: Optional[float] = typer.Option(None, help="Total decomposition timeout in seconds (None = no limit)."),
    qa_timeout: Optional[float] = typer.Option(None, help="Per-LLM-call timeout in seconds (None = no limit)."),
    no_progress: bool = typer.Option(False, help="Disable the rich progress display."),
    verbose: bool = typer.Option(False, help="Show detailed intermediate results."),
) -> None:
    """Run Atom-of-Thought reasoning on a question."""

    async def _run() -> None:
        from chatterer.strategies import (
            AoTTimeoutConfig,
            RichProgressCallback,
            aot_graph,
            aot_input,
            aot_output,
        )

        # Build timeout config
        timeout_config: AoTTimeoutConfig | None = None
        if timeout is not None or qa_timeout is not None:
            timeout_config = AoTTimeoutConfig(
                qa_timeout=qa_timeout,
                decomposition_timeout=timeout,
            )

        # Build progress callback
        progress: RichProgressCallback | None = None
        if not no_progress:
            progress = RichProgressCallback()

        # Build graph
        graph = aot_graph(
            Chatterer.from_provider(chatterer),
            max_depth=max_depth,
            max_sub_questions=max_sub_questions,
            max_workers=max_workers,
            on_progress=progress if progress else None,
        )

        # Prepare input
        state = aot_input(
            question,
            timeout=timeout_config,
        )

        # Run with progress display
        if progress:
            progress.start()
        try:
            raw = cast(AoTState, await graph.ainvoke(state))  # pyright: ignore[reportUnknownMemberType]
        finally:
            if progress:
                progress.stop()

        # Extract result
        result = aot_output(raw)

        # Display
        if verbose:
            result.pretty_print()
        else:
            from rich.console import Console
            from rich.panel import Panel
            from rich.text import Text

            console = Console()

            confidence_color = "green" if result.confidence >= 0.7 else "yellow" if result.confidence >= 0.4 else "red"
            answer_text = Text()
            answer_text.append(result.answer)
            answer_text.append("\n\nConfidence: ", style="dim")
            answer_text.append(f"{result.confidence:.2f}", style=f"bold {confidence_color}")

            console.print(Panel(answer_text, title="[bold cyan]Answer[/]", border_style="cyan"))

            if result.timed_out:
                console.print(
                    Panel(
                        f"Reason: {result.timeout_reason.value}",
                        title="[bold red]Timeout Occurred[/]",
                        border_style="red",
                    )
                )

    asyncio.run(_run())


def main() -> None:
    """Main entry point."""
    try:
        typer.run(command)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        from loguru import logger

        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
