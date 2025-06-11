from __future__ import annotations
import importlib
import sys
from typing import Dict, Tuple, Type

from spargear import BaseArguments, SubcommandSpec


class _Placeholder(BaseArguments):
    """Placeholder subcommand used before lazy import."""
    pass


_SUBCOMMAND_INFO: Dict[str, Tuple[str, str]] = {
    "pdf2md": ("chatterer.examples.pdf_to_markdown", "PdfToMarkdownArgs"),
    "pdf2text": ("chatterer.examples.pdf_to_text", "PdfToTextArgs"),
    "web2md": ("chatterer.examples.webpage_to_markdown", "WebpageToMarkdownArgs"),
    "upstage-parse": ("chatterer.examples.upstage_parser", "UpstageParserArguments"),
    "transcribe": ("chatterer.examples.transcription_api", "TranscriptionApiArguments"),
    "ppt": ("chatterer.examples.make_ppt", "MakePptArguments"),
    "pw-login": ("chatterer.examples.login_with_playwright", "LoginWithPlaywrightArgs"),
    "snippets": ("chatterer.examples.get_code_snippets", "GetCodeSnippetsArgs"),
    "any2md": ("chatterer.examples.anything_to_markdown", "AnythingToMarkdownArguments"),
}


class Arguments(BaseArguments):
    """Chatterer multi-tool command."""

    pdf2md: SubcommandSpec[BaseArguments] = SubcommandSpec("pdf2md", _Placeholder, help="Convert PDF to Markdown")
    pdf2text: SubcommandSpec[BaseArguments] = SubcommandSpec("pdf2text", _Placeholder, help="Convert PDF to text")
    web2md: SubcommandSpec[BaseArguments] = SubcommandSpec("web2md", _Placeholder, help="Convert webpage to Markdown")
    upstage_parse: SubcommandSpec[BaseArguments] = SubcommandSpec("upstage-parse", _Placeholder, help="Parse document with Upstage")
    transcribe: SubcommandSpec[BaseArguments] = SubcommandSpec("transcribe", _Placeholder, help="Transcribe audio")
    ppt: SubcommandSpec[BaseArguments] = SubcommandSpec("ppt", _Placeholder, help="Generate presentation slides")
    pw_login: SubcommandSpec[BaseArguments] = SubcommandSpec("pw-login", _Placeholder, help="Manage Playwright login")
    snippets: SubcommandSpec[BaseArguments] = SubcommandSpec("snippets", _Placeholder, help="Extract code snippets")
    any2md: SubcommandSpec[BaseArguments] = SubcommandSpec("any2md", _Placeholder, help="Convert files to Markdown")

    def __init__(self, args: list[str] | None = None) -> None:
        if args is None:
            args = sys.argv[1:]

        original_subs = self.__class__.__subcommands__.copy()
        if args:
            name = args[0]
            info = _SUBCOMMAND_INFO.get(name)
            spec = self.__class__.__subcommands__.get(name)
            if info and spec:
                module = importlib.import_module(info[0])
                sub_cls: Type[BaseArguments] = getattr(module, info[1])
                sub_cls.__parent__ = self.__class__
                spec.argument_class = sub_cls
                for other in list(self.__class__.__subcommands__.keys()):
                    if other != name:
                        del self.__class__.__subcommands__[other]
        super().__init__(args)
        self.__class__.__subcommands__ = original_subs


def main(argv: list[str] | None = None) -> None:
    args = Arguments(argv)
    sub = args.last_subcommand
    if sub is None:
        Arguments.get_parser().print_help()
        return
    if hasattr(sub, "run"):
        result = sub.run()
        if result is not None:
            print(result)


if __name__ == "__main__":
    main()
