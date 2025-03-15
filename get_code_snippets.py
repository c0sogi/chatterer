from pathlib import Path

from chatterer import pyscripts_to_snippets

if __name__ == "__main__":
    print("Enter a package name or path: ", end="")
    path_or_pkgname: str = input()
    code_snippets = pyscripts_to_snippets(
        path_or_pkgname=path_or_pkgname,
        ban_file_patterns=[".venv/*", Path(__file__).relative_to(Path.cwd()).as_posix()],
    )
    print(
        f"{code_snippets.metadata}\nWould you like to save the result? [y/n]: ",
        end="",
    )
    if input().lower() == "y":
        save_path: Path = Path(__file__).with_suffix(".txt")
        save_path.write_text(code_snippets.snippets_text, encoding="utf-8")
        print(f"Saved to {save_path}")
    else:
        print(code_snippets.snippets_text)
