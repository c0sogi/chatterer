"""
Rich-based progress display for Atom-of-Thought reasoning pipeline.

Provides adaptive terminal layouts and real-time status updates.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


def _detect_unicode_support() -> bool:
    """Detect if the terminal supports Unicode characters."""
    import sys

    try:
        # Check if stdout encoding supports common Unicode characters
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        # Test encoding common special characters
        "\u2713\u2717\u25cb\u22ef".encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


# Symbols for different terminal support levels
_UNICODE_SYMBOLS = {
    "pending": "\u25cb",  # ○
    "running": "\u22ef",  # ⋯
    "complete": "\u2713",  # ✓
    "error": "\u2717",  # ✗
    "arrow": "\u2500\u2500\u2192",  # ──→
    "branch_h": "\u2500",  # ─
    "branch_v": "\u2502",  # │
    "branch_tee": "\u251c",  # ├
    "branch_corner": "\u2514",  # └
    "branch_cross": "\u2534",  # ┴
    "branch_top_left": "\u250c",  # ┌
    "branch_top_right": "\u2510",  # ┐
}

_ASCII_SYMBOLS = {
    "pending": "o",
    "running": "...",
    "complete": "+",
    "error": "x",
    "arrow": "-->",
    "branch_h": "-",
    "branch_v": "|",
    "branch_tee": "|",
    "branch_corner": "`",
    "branch_cross": "+",
    "branch_top_left": "+",
    "branch_top_right": "+",
}


def _get_symbols() -> dict[str, str]:
    """Get appropriate symbol set based on terminal capabilities."""
    if _detect_unicode_support():
        return _UNICODE_SYMBOLS
    return _ASCII_SYMBOLS


class StageStatus(Enum):
    """Status of a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"
    TIMEOUT = "timeout"


class LayoutMode(Enum):
    """Terminal layout modes based on width."""

    FULL = "full"  # 200+ columns
    WIDE = "wide"  # 120-199 columns
    NARROW = "narrow"  # 80-119 columns
    COMPACT = "compact"  # <80 columns


@dataclass
class StageState:
    """State tracking for a single pipeline stage."""

    name: str
    status: StageStatus = StageStatus.PENDING
    message: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    children: list["StageState"] = field(default_factory=lambda: [])

    @property
    def elapsed(self) -> Optional[float]:
        """Calculate elapsed time for this stage."""
        if self.start_time is None:
            return None
        end = self.end_time or time.time()
        return end - self.start_time


@dataclass
class SubQuestionNode:
    """Node in sub-question tree for tracking recursive decomposition."""

    path: str
    question: str = ""
    answer: str = ""
    status: StageStatus = StageStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    children: dict[str, "SubQuestionNode"] = field(default_factory=lambda: {})
    timeout_reason: Optional[str] = None
    active_futures: Optional[int] = None

    @property
    def elapsed(self) -> Optional[float]:
        """Calculate elapsed time for this node."""
        if self.start_time is None:
            return None
        end = self.end_time or time.time()
        return end - self.start_time


# Stage display order and groupings
_MAIN_STAGES = ["prepare", "parallel", "contract", "ensemble"]
_PARALLEL_CHILDREN = ["direct", "decompose"]


def _format_elapsed(seconds: Optional[float]) -> str:
    """Format elapsed time as human-readable string."""
    if seconds is None:
        return ""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m{secs:.1f}s"


def _sanitize_text(text: str) -> str:
    """Replace line breaks and problematic characters for single-line display."""
    return text.replace("\n", " ↵ ").replace("\r", "").replace("\t", " ")


def _status_symbol(status: StageStatus, symbols: Optional[dict[str, str]] = None) -> Text:
    """Get status symbol with appropriate color."""
    if symbols is None:
        symbols = _get_symbols()

    if status == StageStatus.PENDING:
        return Text(symbols["pending"], style="dim")
    elif status == StageStatus.RUNNING:
        return Text(symbols["running"], style="yellow bold")
    elif status == StageStatus.COMPLETE:
        return Text(symbols["complete"], style="green bold")
    elif status == StageStatus.TIMEOUT:
        return Text("⏱", style="orange1 bold")
    else:  # ERROR
        return Text(symbols["error"], style="red bold")


def _status_word(status: StageStatus) -> Text:
    """Get status word with appropriate color."""
    if status == StageStatus.PENDING:
        return Text("pending", style="dim")
    elif status == StageStatus.RUNNING:
        return Text("running", style="yellow")
    elif status == StageStatus.COMPLETE:
        return Text("complete", style="green")
    elif status == StageStatus.TIMEOUT:
        return Text("timeout", style="orange1")
    else:  # ERROR
        return Text("error", style="red")


class RichProgressCallback:
    """
    Rich-based progress callback for AoT pipeline visualization.

    Adapts display layout to terminal width and provides real-time updates.

    Usage:
        with RichProgressCallback() as progress:
            graph = aot_graph(chatterer, on_progress=progress)
            state = aot_input(question)
            raw = aot_run_sync(graph, state)
            result = aot_output(raw)

        # Or manually:
        progress = RichProgressCallback(auto_start=False)
        progress.start()
        graph = aot_graph(chatterer, on_progress=progress)
        state = aot_input(question)
        raw = aot_run_sync(graph, state)
        result = aot_output(raw)
        progress.stop()
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        refresh_per_second: float = 4,
        auto_start: bool = True,
    ) -> None:
        """
        Initialize the progress callback.

        Args:
            console: Rich Console instance. Created if not provided.
            refresh_per_second: Live display refresh rate.
            auto_start: Whether to auto-start when used as context manager.
        """
        self._console = console or Console()
        self._refresh_per_second = refresh_per_second
        self._auto_start = auto_start

        self._lock = threading.Lock()
        self._live: Optional[Live] = None
        self._started = False

        # Cache symbols for consistent use
        self._symbols = _get_symbols()

        # Stage states keyed by name
        self._stages: dict[str, StageState] = {}

        # Sub-question tree (root node for decomposition)
        self._subq_root: SubQuestionNode = SubQuestionNode(path="root")

        # Legacy sub-questions list for backward compatibility
        self._sub_questions: list[StageState] = []

        # Initialize main stages
        for stage_name in _MAIN_STAGES:
            self._stages[stage_name] = StageState(name=stage_name)

        # Initialize parallel children
        for child_name in _PARALLEL_CHILDREN:
            self._stages[child_name] = StageState(name=child_name)

    def __call__(self, stage: str, message: str) -> None:
        """
        Handle progress callback from AoT pipeline.

        This is the ProgressCallback signature expected by aot_graph.

        Args:
            stage: Stage name (prepare, parallel, decompose, direct, contract, ensemble)
            message: Progress message
        """
        with self._lock:
            self._update_stage(stage, message)
            if self._live:
                self._live.update(self._render())

    def __enter__(self) -> RichProgressCallback:
        """Start the Live display when entering context."""
        if self._auto_start:
            self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop the Live display when exiting context."""
        self.stop()

    def start(self) -> None:
        """Manually start the Live display."""
        if self._started:
            return

        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=self._refresh_per_second,
            transient=False,
        )
        self._live.start()
        self._started = True

    def stop(self) -> None:
        """Manually stop the Live display."""
        if not self._started or self._live is None:
            return

        # Final render before stopping
        self._live.update(self._render())
        self._live.stop()
        self._live = None
        self._started = False

    def _mark_completed_stages(self, current_stage: str) -> None:
        """Mark prior stages as complete based on stage transitions.

        Only marks stages complete when moving to specific later stages,
        not aggressively on any message.
        """
        # Only mark prepare complete when parallel starts
        if current_stage == "parallel":
            prepare = self._stages.get("prepare")
            if prepare and prepare.status != StageStatus.COMPLETE:
                prepare.status = StageStatus.COMPLETE
                if prepare.end_time is None:
                    prepare.end_time = time.time()

        # Only mark parallel complete when contract starts
        if current_stage == "contract":
            parallel = self._stages.get("parallel")
            if parallel and parallel.status != StageStatus.COMPLETE:
                parallel.status = StageStatus.COMPLETE
                if parallel.end_time is None:
                    parallel.end_time = time.time()

        # Only mark contract complete when ensemble starts
        if current_stage == "ensemble":
            contract = self._stages.get("contract")
            if contract and contract.status != StageStatus.COMPLETE:
                contract.status = StageStatus.COMPLETE
                if contract.end_time is None:
                    contract.end_time = time.time()

    def _get_subq_node(self, path: str) -> Optional[SubQuestionNode]:
        """Get a sub-question node by path (e.g., 'root.0.1')."""
        if path == "root":
            return self._subq_root

        parts = path.split(".")
        if parts[0] != "root":
            return None

        node = self._subq_root
        for part in parts[1:]:
            if part not in node.children:
                return None
            node = node.children[part]
        return node

    def _ensure_subq_node(self, path: str) -> SubQuestionNode:
        """Ensure a sub-question node exists at path, creating parents as needed."""
        if path == "root":
            return self._subq_root

        parts = path.split(".")
        node = self._subq_root
        for i, part in enumerate(parts[1:], start=1):
            if part not in node.children:
                child_path = ".".join(parts[: i + 1])
                node.children[part] = SubQuestionNode(path=child_path)
            node = node.children[part]
        return node

    def _update_stage(self, stage: str, message: str) -> None:
        """Update stage state based on incoming progress message."""
        # Infer completion from stage transitions
        self._mark_completed_stages(stage)

        # Handle sub-question messages with new format: ACTION|path|details
        if stage == "subq":
            self._handle_subq_message(message)
            return

        # Handle decompose messages with new format: ACTION|key=value|...
        if stage == "decompose":
            self._handle_decompose_message(message)
            return

        # Parse stage and detect status from message
        stage_state = self._stages.get(stage)

        if stage_state is None:
            # Unknown stage, create it
            stage_state = StageState(name=stage)
            self._stages[stage] = stage_state

        # Handle explicit COMPLETE signal
        if message == "COMPLETE":
            stage_state.status = StageStatus.COMPLETE
            if stage_state.end_time is None:
                stage_state.end_time = time.time()
            stage_state.message = "Complete"
            return

        # Detect status from message patterns (backward compatibility)
        if "Done" in message or "complete" in message.lower():
            stage_state.status = StageStatus.COMPLETE
            if stage_state.end_time is None:
                stage_state.end_time = time.time()
        elif stage_state.status == StageStatus.PENDING:
            stage_state.status = StageStatus.RUNNING
            if stage_state.start_time is None:
                stage_state.start_time = time.time()

        stage_state.message = message

        # Propagate parallel status to children
        if stage == "parallel":
            for child_name in _PARALLEL_CHILDREN:
                child = self._stages.get(child_name)
                if child and child.status == StageStatus.PENDING:
                    child.status = StageStatus.RUNNING
                    child.start_time = time.time()

    def _handle_subq_message(self, message: str) -> None:
        """Handle sub-question messages with format: ACTION|path|details."""
        parts = message.split("|")
        action = parts[0] if parts else ""
        path = parts[1] if len(parts) > 1 else ""
        details = parts[2] if len(parts) > 2 else ""

        if action == "NEW":
            # Create new sub-question node with question text
            node = self._ensure_subq_node(path)
            node.question = details
            node.status = StageStatus.PENDING
            # Also add to legacy list for backward compatibility rendering
            self._add_legacy_subq(path, details)

        elif action == "START":
            # Mark node as running
            node = self._get_subq_node(path)
            if node:
                node.status = StageStatus.RUNNING
                node.start_time = time.time()
            self._update_legacy_subq_status(path, StageStatus.RUNNING)

        elif action == "DONE":
            # Mark node as complete, optionally with answer
            node = self._get_subq_node(path)
            if node:
                node.status = StageStatus.COMPLETE
                node.end_time = time.time()
                if details:
                    node.answer = details
            self._update_legacy_subq_status(path, StageStatus.COMPLETE)

        elif action == "ERROR":
            # Mark node as error
            node = self._get_subq_node(path)
            if node:
                node.status = StageStatus.ERROR
                node.end_time = time.time()
            self._update_legacy_subq_status(path, StageStatus.ERROR)

        elif action == "TIMEOUT":
            # Format: TIMEOUT|path|reason|active_futures=N
            # OR: TIMEOUT|path={path}|reason (for decompose)
            # Extract path from either format
            actual_path = path
            reason = details if details else "unknown"
            active_futures = 0

            # Handle path={value} format
            if path.startswith("path="):
                actual_path = path.split("=", 1)[1]
                reason = details if details else "unknown"
            else:
                # Standard format: TIMEOUT|path|reason|active_futures=N
                if len(parts) > 3 and "active_futures=" in parts[3]:
                    try:
                        active_futures = int(parts[3].split("=")[1])
                    except (ValueError, IndexError):
                        pass

            node = self._get_subq_node(actual_path)
            if node:
                node.status = StageStatus.TIMEOUT
                node.end_time = time.time()
                node.timeout_reason = reason
                node.active_futures = active_futures
            self._update_legacy_subq_status(actual_path, StageStatus.TIMEOUT)

        elif action == "LEAF":
            # Leaf node (direct answer at max depth)
            node = self._ensure_subq_node(path)
            node.question = details
            node.status = StageStatus.RUNNING
            node.start_time = time.time()

    def _handle_decompose_message(self, message: str) -> None:
        """Handle decompose messages with format: ACTION|key=value|..."""
        decompose_state = self._stages.get("decompose")
        if decompose_state is None:
            decompose_state = StageState(name="decompose")
            self._stages["decompose"] = decompose_state

        # Handle explicit COMPLETE signal
        if message == "COMPLETE":
            decompose_state.status = StageStatus.COMPLETE
            if decompose_state.end_time is None:
                decompose_state.end_time = time.time()
            decompose_state.message = "Complete"
            # Mark all sub-questions as complete
            self._mark_all_subq_complete()
            return

        parts = message.split("|")
        action = parts[0] if parts else ""

        if action == "DECOMPOSE":
            # Starting decomposition at a depth/path
            if decompose_state.status == StageStatus.PENDING:
                decompose_state.status = StageStatus.RUNNING
                decompose_state.start_time = time.time()
            # Extract depth for display
            for part in parts[1:]:
                if part.startswith("depth="):
                    depth = part.split("=")[1]
                    decompose_state.message = f"Decomposing at depth {depth}..."

        elif action == "SYNTHESIZED":
            # Synthesis complete at a path (intermediate step)
            decompose_state.message = "Synthesizing answers..."

        else:
            # Backward compatibility: old message format
            decompose_state.message = message
            if decompose_state.status == StageStatus.PENDING:
                decompose_state.status = StageStatus.RUNNING
                decompose_state.start_time = time.time()

            # Legacy: parse "Generated N sub-questions"
            if "Generated" in message:
                try:
                    msg_parts = message.split()
                    idx = msg_parts.index("Generated")
                    if idx + 1 < len(msg_parts):
                        count = int(msg_parts[idx + 1])
                        self._sub_questions = [
                            StageState(name=f"Q{i + 1}", status=StageStatus.RUNNING, start_time=time.time())
                            for i in range(count)
                        ]
                except (ValueError, IndexError):
                    pass

    def _add_legacy_subq(self, path: str, question: str) -> None:
        """Add to legacy sub-questions list for backward compatible rendering."""
        # Extract index from path like "root.0" or "root.1"
        parts = path.split(".")
        if len(parts) >= 2:
            try:
                idx = int(parts[-1])
                # Ensure list is large enough
                while len(self._sub_questions) <= idx:
                    self._sub_questions.append(
                        StageState(name=f"Q{len(self._sub_questions) + 1}", status=StageStatus.PENDING)
                    )
                # Update with question text (truncated for display)
                display_text = question[:50] + "..." if len(question) > 50 else question
                self._sub_questions[idx].message = display_text
            except ValueError:
                pass

    def _update_legacy_subq_status(self, path: str, status: StageStatus) -> None:
        """Update legacy sub-question status by path."""
        parts = path.split(".")
        if len(parts) >= 2:
            try:
                idx = int(parts[-1])
                if idx < len(self._sub_questions):
                    self._sub_questions[idx].status = status
                    if status == StageStatus.RUNNING:
                        self._sub_questions[idx].start_time = time.time()
                    elif status in (StageStatus.COMPLETE, StageStatus.ERROR):
                        self._sub_questions[idx].end_time = time.time()
            except ValueError:
                pass

    def _mark_all_subq_complete(self) -> None:
        """Mark all sub-questions as complete."""
        for sq in self._sub_questions:
            if sq.status == StageStatus.RUNNING:
                sq.status = StageStatus.COMPLETE
                sq.end_time = time.time()

        def mark_tree_complete(node: SubQuestionNode) -> None:
            if node.status == StageStatus.RUNNING:
                node.status = StageStatus.COMPLETE
                node.end_time = time.time()
            for child in node.children.values():
                mark_tree_complete(child)

        mark_tree_complete(self._subq_root)

    def _get_layout_mode(self) -> LayoutMode:
        """Determine layout mode based on terminal width."""
        width = self._console.width
        if width >= 200:
            return LayoutMode.FULL
        elif width >= 120:
            return LayoutMode.WIDE
        elif width >= 80:
            return LayoutMode.NARROW
        else:
            return LayoutMode.COMPACT

    def _render(self) -> RenderableType:
        """Render the progress display based on current layout mode."""
        mode = self._get_layout_mode()

        if mode == LayoutMode.COMPACT:
            return self._render_compact()
        elif mode == LayoutMode.NARROW:
            return self._render_narrow()
        elif mode == LayoutMode.WIDE:
            return self._render_wide()
        else:  # FULL
            return self._render_full()

    def _render_compact(self) -> RenderableType:
        """Render minimal status line for very narrow terminals."""
        # Find current stage
        current_stage = None
        stage_num = 0
        for i, name in enumerate(_MAIN_STAGES):
            stage = self._stages.get(name)
            if stage and stage.status == StageStatus.RUNNING:
                current_stage = stage
                stage_num = i + 1
                break
            elif stage and stage.status == StageStatus.COMPLETE:
                stage_num = i + 1

        line1 = Text()
        line1.append("AoT ", style="bold cyan")
        line1.append(f"[{stage_num}/{len(_MAIN_STAGES)}] ", style="dim")
        if current_stage:
            line1.append(current_stage.name, style="yellow")
            line1.append(" ")
            line1.append(_status_symbol(current_stage.status, self._symbols))

        # Second line: parallel children status
        direct = self._stages.get("direct")
        decompose = self._stages.get("decompose")

        line2 = Text("  ")
        if direct:
            line2.append("direct: ")
            line2.append(_status_symbol(direct.status, self._symbols))
        line2.append("  ")
        if decompose:
            line2.append("decompose: ")
            line2.append(_status_symbol(decompose.status, self._symbols))
            if self._sub_questions:
                line2.append(f" ({len(self._sub_questions)} subs)", style="dim")

        return Group(line1, line2)

    def _render_narrow(self) -> RenderableType:
        """Render vertical list with indented children."""
        content: list[RenderableType] = []

        for i, name in enumerate(_MAIN_STAGES):
            stage = self._stages.get(name)
            if not stage:
                continue

            # Main stage line
            line = Text()
            line.append(f"[{i + 1}] ", style="dim")
            line.append(f"{name} ", style="bold" if stage.status == StageStatus.RUNNING else None)
            line.append("." * (20 - len(name)), style="dim")
            line.append(" ")
            line.append(_status_symbol(stage.status, self._symbols))
            line.append(" ")
            if stage.status == StageStatus.RUNNING:
                line.append(_status_word(stage.status))
            elif stage.elapsed is not None:
                line.append(f"({_format_elapsed(stage.elapsed)})", style="dim")

            content.append(line)

            # Children for parallel stage
            if name == "parallel":
                sym = self._symbols
                for child_name in _PARALLEL_CHILDREN:
                    child = self._stages.get(child_name)
                    if not child:
                        continue

                    is_last_child = child_name == _PARALLEL_CHILDREN[-1] and not self._sub_questions
                    prefix = (
                        f"{sym['branch_corner']}{sym['branch_h']}{sym['branch_h']} "
                        if is_last_child
                        else f"{sym['branch_tee']}{sym['branch_h']}{sym['branch_h']} "
                    )

                    child_line = Text()
                    child_line.append("    ", style="dim")
                    child_line.append(prefix, style="dim")
                    child_line.append(f"{child_name} ", style="bold" if child.status == StageStatus.RUNNING else None)
                    child_line.append("." * (14 - len(child_name)), style="dim")
                    child_line.append(" ")
                    child_line.append(_status_symbol(child.status, self._symbols))
                    child_line.append(" ")
                    if child.status == StageStatus.RUNNING:
                        child_line.append(_status_word(child.status))
                    elif child.elapsed is not None:
                        child_line.append(f"({_format_elapsed(child.elapsed)})", style="dim")

                    content.append(child_line)

                    # Sub-questions under decompose - render as simple tree for narrow layout
                    if child_name == "decompose" and self._subq_root.children:
                        nodes = list(self._subq_root.children.values())
                        for i, node in enumerate(nodes):
                            is_last_node = i == len(nodes) - 1
                            subq_lines = self._render_nested_subq(node, "        ", is_last_node)
                            content.extend(subq_lines)

        return Panel(
            Group(*content),
            title="[bold cyan]AoT Pipeline[/]",
            border_style="cyan",
            padding=(0, 1),
        )

    def _render_wide(self) -> RenderableType:
        """Render horizontal pipeline with vertical tree."""
        sym = self._symbols

        # Top: horizontal pipeline
        pipeline = Text()
        pipeline.append("  ")

        for i, name in enumerate(_MAIN_STAGES):
            stage = self._stages.get(name)
            if not stage:
                continue

            # Stage box
            if stage.status == StageStatus.RUNNING:
                pipeline.append(f"[{name}]", style="yellow bold")
            elif stage.status == StageStatus.COMPLETE:
                pipeline.append(f"[{name}]", style="green")
            elif stage.status == StageStatus.ERROR:
                pipeline.append(f"[{name}]", style="red")
            else:
                pipeline.append(f"[{name}]", style="dim")

            # Arrow between stages (except last)
            if i < len(_MAIN_STAGES) - 1:
                pipeline.append(f" {sym['arrow']} ", style="dim")

        # Status line under pipeline
        status_line = Text()
        status_line.append("    ")
        for i, name in enumerate(_MAIN_STAGES):
            stage = self._stages.get(name)
            if not stage:
                continue

            if stage.status == StageStatus.RUNNING:
                status_text = f"{sym['running']} running"
            elif stage.status == StageStatus.COMPLETE and stage.elapsed:
                status_text = f"{sym['complete']} {_format_elapsed(stage.elapsed)}"
            elif stage.status == StageStatus.PENDING:
                status_text = sym["pending"]
            else:
                status_text = ""

            # Pad to align with stage name (approximate)
            base_len = len(
                status_text.replace(sym["running"], " ").replace(sym["complete"], " ").replace(sym["pending"], " ")
            )
            padding = len(name) + 2 - base_len
            status_line.append(status_text)
            if i < len(_MAIN_STAGES) - 1:
                status_line.append(" " * max(1, padding + 5))

        # Parallel branch visualization
        branch_content: list[RenderableType] = []

        parallel = self._stages.get("parallel")
        if parallel and parallel.status != StageStatus.PENDING:
            branch_content.append(Text(""))
            branch_content.append(Text(f"          {sym['branch_v']}", style="dim"))
            branch_content.append(
                Text(
                    f"  {sym['branch_top_left']}{sym['branch_h'] * 7}{sym['branch_cross']}{sym['branch_h'] * 7}{sym['branch_top_right']}",
                    style="dim",
                )
            )
            branch_content.append(Text(f"  {sym['branch_v']}               {sym['branch_v']}", style="dim"))

            # Direct and decompose
            direct = self._stages.get("direct")
            decompose = self._stages.get("decompose")

            direct_text = Text("[direct]", style=self._stage_style(direct) if direct else "dim")
            decompose_text = Text("[decompose]", style=self._stage_style(decompose) if decompose else "dim")

            branch_line = Text()
            branch_line.append("  ")
            branch_line.append(direct_text)
            branch_line.append("    ")
            branch_line.append(decompose_text)
            branch_content.append(branch_line)

            # Timing line - check status FIRST, then elapsed
            timing_line = Text()
            timing_line.append("   ")
            if direct:
                if direct.status == StageStatus.COMPLETE:
                    elapsed_str = _format_elapsed(direct.elapsed) if direct.elapsed else ""
                    timing_line.append(f"{sym['complete']} {elapsed_str}", style="green")
                elif direct.status == StageStatus.RUNNING:
                    elapsed_str = _format_elapsed(direct.elapsed) if direct.elapsed else "..."
                    timing_line.append(f"{sym['running']} {elapsed_str}", style="yellow")
                else:
                    timing_line.append(sym["pending"], style="dim")
            else:
                timing_line.append("       ")
            timing_line.append("       ")
            if decompose:
                if decompose.status == StageStatus.COMPLETE:
                    elapsed_str = _format_elapsed(decompose.elapsed) if decompose.elapsed else ""
                    timing_line.append(f"{sym['complete']} {elapsed_str}", style="green")
                elif decompose.status == StageStatus.RUNNING:
                    elapsed_str = _format_elapsed(decompose.elapsed) if decompose.elapsed else "..."
                    timing_line.append(f"{sym['running']} {elapsed_str}", style="yellow")
                else:
                    timing_line.append(sym["pending"], style="dim")
            branch_content.append(timing_line)

            # Sub-questions as individual panels
            if self._subq_root.children:
                branch_content.append(Text(""))
                branch_content.append(Text("          Sub-Questions:", style="bold"))
                subq_panels = self._render_subq_panels()
                branch_content.extend(subq_panels)

        return Panel(
            Group(pipeline, status_line, *branch_content),
            title="[bold cyan]AoT Reasoning Pipeline[/]",
            border_style="cyan",
            padding=(1, 2),
        )

    def _render_full(self) -> RenderableType:
        """Render full graph with maximum detail (200+ columns)."""
        # Use wide layout as base, but with more spacing and detail
        return self._render_wide()

    def _stage_style(self, stage: Optional[StageState]) -> str:
        """Get style string for a stage based on status."""
        if stage is None:
            return "dim"
        if stage.status == StageStatus.RUNNING:
            return "yellow bold"
        elif stage.status == StageStatus.COMPLETE:
            return "green"
        elif stage.status == StageStatus.ERROR:
            return "red"
        else:
            return "dim"

    def _render_subq_panels(self) -> list[RenderableType]:
        """Render first-level sub-questions as individual panels with nested trees."""
        panels: list[RenderableType] = []
        sym = self._symbols

        for node in self._subq_root.children.values():
            # Build panel content
            content_lines: list[Text] = []

            # Width calculation for panel content:
            # - Outer pipeline panel: borders(2) + padding(4) = 6
            # - This subquestion panel: borders(2) + padding(4) = 6
            # - "→ " prefix (2)
            # - Safety margin (4)
            # Total overhead: ~18 chars
            panel_content_width = max(25, self._console.width - 20)

            # Answer line (if available)
            if node.answer:
                answer_clean = _sanitize_text(node.answer)
                max_a_len = panel_content_width - 2  # account for "→ "
                answer_text = answer_clean[:max_a_len] + "..." if len(answer_clean) > max_a_len else answer_clean
                answer_line = Text()
                answer_line.append("→ ", style="dim")
                answer_line.append(answer_text, style="green" if node.status == StageStatus.COMPLETE else None)
                content_lines.append(answer_line)
            elif node.status == StageStatus.RUNNING:
                content_lines.append(Text("→ Processing...", style="yellow"))

            # Nested children as tree
            if node.children:
                children = list(node.children.values())
                for i, child in enumerate(children):
                    is_last = i == len(children) - 1
                    child_lines = self._render_nested_subq(child, "  ", is_last)
                    content_lines.extend(child_lines)

            # Build title with question (sanitize and truncate)
            # Title has markup overhead + outer panel overhead
            max_q_len = max(20, self._console.width - 30)
            question_clean = _sanitize_text(node.question) if node.question else "Processing..."
            title_q = question_clean[:max_q_len] + "..." if len(question_clean) > max_q_len else question_clean

            # Build subtitle with status and timing
            status_sym = _status_symbol(node.status, sym)
            elapsed = _format_elapsed(node.elapsed) if node.elapsed else "..."
            subtitle = Text()
            subtitle.append(status_sym)
            subtitle.append(f" {elapsed}", style="dim")

            # Determine border style based on status
            if node.status == StageStatus.COMPLETE:
                border_style = "green"
            elif node.status == StageStatus.RUNNING:
                border_style = "yellow"
            elif node.status == StageStatus.ERROR:
                border_style = "red"
            elif node.status == StageStatus.TIMEOUT:
                border_style = "orange1"
            else:
                border_style = "dim"

            panel = Panel(
                Group(*content_lines) if content_lines else Text("Starting...", style="dim"),
                title=f"[bold]{title_q}[/]",
                subtitle=subtitle,
                border_style=border_style,
                padding=(0, 2),  # increased horizontal padding
            )
            panels.append(panel)

        return panels

    def _render_nested_subq(self, node: SubQuestionNode, continuation: str = "", is_last: bool = True) -> list[Text]:
        """Render nested sub-questions as tree with connected pipes."""
        lines: list[Text] = []
        sym = self._symbols

        # Build prefix: continuation pipes + current branch character
        branch_char = sym["branch_corner"] if is_last else sym["branch_tee"]
        prefix = continuation + f"{branch_char}{sym['branch_h']}{sym['branch_h']} "

        line = Text()
        line.append(prefix, style="dim")

        # Calculate available width accounting for nested panel structure:
        # - Outer pipeline panel: borders(2) + padding(4) = 6
        # - Inner subquestion panel: borders(2) + padding(4) = 6
        # - prefix (continuation + branch chars)
        # - suffix: " " + status_symbol(2) + " " + elapsed(10) = ~15
        # Total panel overhead: ~27 chars
        prefix_len = len(continuation) + 4  # branch chars
        panel_overhead = 30  # nested panels + status + elapsed + safety margin
        max_len = max(15, self._console.width - prefix_len - panel_overhead)

        if node.status == StageStatus.COMPLETE and node.answer:
            # Show Q → A format (sanitize both)
            q_text = _sanitize_text(node.question) if node.question else "?"
            a_text = _sanitize_text(node.answer)
            q_max = min(len(q_text), max_len // 3)
            a_max = max_len - q_max - 4  # " → " takes 3 chars
            display_q = q_text[:q_max] + "..." if len(q_text) > q_max else q_text
            display_a = a_text[:a_max] + "..." if len(a_text) > a_max else a_text
            line.append(f"{display_q} → {display_a}", style="green")
        elif node.status == StageStatus.TIMEOUT:
            # Show timeout with reason
            question = _sanitize_text(node.question) if node.question else "Processing..."
            display_q = question[:max_len] + "..." if len(question) > max_len else question
            line.append(display_q, style="dim")
            line.append(" [orange1]⏱ TIMEOUT[/orange1]")
            if node.timeout_reason:
                line.append(f" ({node.timeout_reason})")
            if node.active_futures and node.active_futures > 0:
                line.append(f" [dim](+{node.active_futures} still running)[/dim]")
        else:
            # Sanitize question
            question = _sanitize_text(node.question) if node.question else "Processing..."
            display_q = question[:max_len] + "..." if len(question) > max_len else question
            line.append(display_q, style="bold" if node.status == StageStatus.RUNNING else None)

        line.append(" ")
        line.append(_status_symbol(node.status, self._symbols))

        if node.elapsed is not None:
            line.append(f" {_format_elapsed(node.elapsed)}", style="dim")

        lines.append(line)

        # Build continuation for children: add pipe if not last, else spaces
        child_continuation = continuation + (f"{sym['branch_v']}   " if not is_last else "    ")

        # Render children recursively
        children = list(node.children.values())
        for i, child in enumerate(children):
            child_is_last = i == len(children) - 1
            lines.extend(self._render_nested_subq(child, child_continuation, child_is_last))

        return lines


__all__ = [
    "StageStatus",
    "StageState",
    "SubQuestionNode",
    "LayoutMode",
    "RichProgressCallback",
]
