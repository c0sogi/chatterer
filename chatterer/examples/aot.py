"""
Atom-of-Thought (AoT) Reasoning CLI

A command-line tool for multi-path reasoning using the AoT pipeline.
Decomposes questions, answers them in parallel, and synthesizes a final answer.
"""

import asyncio
import sys
from typing import Optional, cast

from spargear import ArgumentSpec, RunnableArguments

from chatterer import Chatterer
from chatterer.constants import DEFAULT_GOOGLE_MODEL
from chatterer.strategies.atom_of_thoughts import AoTState


class Arguments(RunnableArguments[None]):
    """Command-line arguments for Atom-of-Thought reasoning."""

    QUESTION: str
    """The question to reason about."""

    chatterer: ArgumentSpec[Chatterer] = ArgumentSpec(
        ["--chatterer"],
        default_factory=lambda: Chatterer.from_provider(f"google:{DEFAULT_GOOGLE_MODEL}"),
        help=f"Chatterer instance configuration (e.g., 'google:{DEFAULT_GOOGLE_MODEL}').",
        type=Chatterer.from_provider,
    )

    max_depth: int = 2
    """Maximum recursion depth for decomposition."""

    max_sub_questions: int = 3
    """Maximum sub-questions per decomposition level."""

    max_workers: int = 4
    """Maximum parallel workers for sub-question processing."""

    timeout: Optional[float] = None
    """Total decomposition timeout in seconds (None = no limit)."""

    qa_timeout: Optional[float] = None
    """Per-LLM-call timeout in seconds (None = no limit)."""

    no_progress: bool = False
    """Disable the rich progress display."""

    verbose: bool = False
    """Show detailed intermediate results."""

    def run(self) -> None:
        """Execute the AoT reasoning pipeline."""
        return asyncio.run(self.arun())

    async def arun(self) -> None:
        """Async execution of AoT reasoning."""
        from chatterer.strategies import (
            AoTTimeoutConfig,
            RichProgressCallback,
            aot_graph,
            aot_input,
            aot_output,
        )

        # Build timeout config
        timeout_config: AoTTimeoutConfig | None = None
        if self.timeout is not None or self.qa_timeout is not None:
            timeout_config = AoTTimeoutConfig(
                qa_timeout=self.qa_timeout,
                decomposition_timeout=self.timeout,
            )

        # Build progress callback
        progress: RichProgressCallback | None = None
        if not self.no_progress:
            progress = RichProgressCallback()

        # Build graph
        graph = aot_graph(
            self.chatterer.unwrap(),
            max_depth=self.max_depth,
            max_sub_questions=self.max_sub_questions,
            max_workers=self.max_workers,
            on_progress=progress if progress else None,
        )

        # Prepare input
        state = aot_input(
            self.QUESTION,
            timeout=timeout_config,
        )

        # Run with progress display
        if progress:
            progress.start()
        try:
            raw = cast(AoTState, await graph.ainvoke(state))  # type: ignore[reportUnknownMemberType]
        finally:
            if progress:
                progress.stop()

        # Extract result
        result = aot_output(raw)

        # Display
        if self.verbose:
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


def main() -> None:
    """Main entry point."""
    try:
        Arguments().run()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        from loguru import logger

        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
