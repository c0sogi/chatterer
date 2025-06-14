from spargear import SubcommandArguments, SubcommandSpec


def any2md():
    from .any2md import Arguments

    return Arguments


def pdf2md():
    from .pdf2md import Arguments

    return Arguments


def pdf2txt():
    from .pdf2txt import Arguments

    return Arguments


def ppt():
    from .ppt import Arguments

    return Arguments


def pw():
    from .pw import Arguments

    return Arguments


def snippet():
    from .snippet import Arguments

    return Arguments


def transcribe():
    from .transcribe import Arguments

    return Arguments


def upstage():
    from .upstage import Arguments

    return Arguments


def web2md():
    from .web2md import Arguments

    return Arguments


class Arguments(SubcommandArguments):
    any2md = SubcommandSpec(name="any2md", argument_class_factory=any2md)
    pdf2md = SubcommandSpec(name="pdf2md", argument_class_factory=pdf2md)
    pdf2txt = SubcommandSpec(name="pdf2txt", argument_class_factory=pdf2txt)
    ppt = SubcommandSpec(name="ppt", argument_class_factory=ppt)
    pw = SubcommandSpec(name="pw", argument_class_factory=pw)
    snippet = SubcommandSpec(name="snippet", argument_class_factory=snippet)
    transcribe = SubcommandSpec(name="transcribe", argument_class_factory=transcribe)
    upstage = SubcommandSpec(name="upstage", argument_class_factory=upstage)
    web2md = SubcommandSpec(name="web2md", argument_class_factory=web2md)


def main():
    Arguments().execute()


if __name__ == "__main__":
    main()
