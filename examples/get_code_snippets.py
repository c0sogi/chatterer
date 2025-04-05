import argparse
import sys
from pathlib import Path

sys.path.append(".")

from chatterer.tools.convert_to_text import CodeSnippets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract code snippets from a package or file and save them to a text file."
    )
    parser.add_argument(
        "path_or_pkgname",
        type=str,
        help="Path to the package or file from which to extract code snippets.",
    )
    parser.add_argument(
        "--ban-file-patterns",
        type=str,
        nargs="*",
        default=[".venv/*", Path(__file__).relative_to(Path.cwd()).as_posix()],
        help="List of file patterns to ignore.",
    )
    parser.add_argument(
        "--glob-patterns",
        type=str,
        nargs="*",
        default=["*.py"],
        help="List of glob patterns to include.",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Enable case-sensitive matching for glob patterns.",
        default=False,
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the result to a file.",
        default=True,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_suffix(".txt"),
        help="Path to save the code snippets text file.",
    )
    args = parser.parse_args()
    code_snippets = CodeSnippets.from_path_or_pkgname(
        path_or_pkgname=args.path_or_pkgname,
        ban_file_patterns=args.ban_file_patterns,
        glob_patterns=args.glob_patterns,
        case_sensitive=args.case_sensitive,
    )
    if args.save:
        if args.out.exists():
            print(f"[*] File {args.out} already exists. Overwriting...")
        else:
            print(f"[*] Saving to {args.out}...")
        save_path: Path = Path(__file__).with_suffix(".txt")
        args.out.write_text(code_snippets.snippets_text, encoding="utf-8")
        print(f"[*] Saved to {save_path}")
    else:
        print(f"[*] Code snippets extracted from {args.path_or_pkgname}:")
        print(code_snippets.snippets_text)
