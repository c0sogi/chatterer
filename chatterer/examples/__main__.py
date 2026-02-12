import ast
import importlib
import pkgutil
from pathlib import Path
from typing import Any, cast

import click
import typer


def _discover_subcommands() -> dict[str, tuple[str, str]]:
    """Auto-discover subcommands from chatterer.examples package.

    Scans all .py modules in chatterer/examples/ (excluding __init__, __main__).
    Uses ast to detect exported `command` functions or `app` Typer instances,
    and extracts the module docstring as help text — all without importing.

    Returns:
        {name: (import_path, help_text)} for each discovered subcommand.
    """
    pkg_dir = Path(__file__).parent
    subcommands: dict[str, tuple[str, str]] = {}

    for info in pkgutil.iter_modules([str(pkg_dir)]):
        if info.name.startswith("_"):
            continue

        module_file = pkg_dir / f"{info.name}.py"
        if not module_file.exists():
            continue

        try:
            tree = ast.parse(module_file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

        # Extract module docstring for help text
        help_text = ast.get_docstring(tree) or f"{info.name} command."
        # Use first line only
        help_text = help_text.strip().split("\n")[0]

        # Detect exported attribute: `command` function or `app` assignment
        attr = _detect_cli_attribute(tree)
        if attr is None:
            continue

        subcommands[info.name] = (f"chatterer.examples.{info.name}:{attr}", help_text)

    return subcommands


def _detect_cli_attribute(tree: ast.Module) -> str | None:
    """Detect if module exports a `command` function or `app` Typer variable."""
    has_command = False
    has_app = False

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "command":
            has_command = True
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "app":
                    has_app = True

    # Prefer `app` (Typer sub-app) over `command` (single function)
    if has_app:
        return "app"
    if has_command:
        return "command"
    return None


class LazyGroup(click.Group):
    """Click group with lazy-loaded subcommands for optional dependencies."""

    def __init__(self, *args: Any, lazy_subcommands: dict[str, tuple[str, str]] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lazy_subcommands: dict[str, tuple[str, str]] = lazy_subcommands or {}

    def list_commands(self, ctx: click.Context) -> list[str]:
        base = super().list_commands(ctx)
        lazy = sorted(self.lazy_subcommands.keys())
        return base + lazy

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        if cmd_name in self.lazy_subcommands:
            return self._lazy_load(cmd_name)
        return super().get_command(ctx, cmd_name)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        commands: list[tuple[str, str]] = []
        for subcommand in self.list_commands(ctx):
            if subcommand in self.lazy_subcommands:
                _, help_text = self.lazy_subcommands[subcommand]
                commands.append((subcommand, help_text))
            else:
                cmd = self.get_command(ctx, subcommand)
                if cmd is None:
                    continue
                help_str: str = cmd.get_short_help_str(limit=150)
                commands.append((subcommand, help_str))
        if commands:
            with formatter.section("Commands"):
                formatter.write_dl(commands)

    def _lazy_load(self, cmd_name: str) -> click.Command:
        import_path, _ = self.lazy_subcommands[cmd_name]
        module_path, attr_name = import_path.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        obj = getattr(mod, attr_name)
        if isinstance(obj, typer.Typer):
            group = typer.main.get_command(obj)
            group.name = cmd_name
            return group
        # It's a bare function - wrap in typer to get the click command
        tmp_app = typer.Typer()
        tmp_app.command(name=cmd_name)(obj)
        tmp_group = typer.main.get_command(tmp_app)
        if isinstance(tmp_group, click.Group):
            cmd = tmp_group.commands.get(cmd_name)
            if cmd:
                return cmd
        return cast(click.Command, tmp_group)


@click.group(cls=LazyGroup, lazy_subcommands=_discover_subcommands())
def main() -> None:
    """The highest-level interface for various LLM APIs."""
    pass


if __name__ == "__main__":
    main()
