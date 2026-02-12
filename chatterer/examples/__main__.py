from spargear import SubcommandArguments, SubcommandSpec


def aot():
    from .aot import Arguments

    return Arguments


def any2md():
    from .any2md import Arguments

    return Arguments


def pdf2md():
    from .pdf2md import Arguments

    return Arguments


def pdf2txt():
    from .pdf2txt import Arguments

    return Arguments


def makeppt():
    from .makeppt import Arguments

    return Arguments


def ppt2pdf():
    from .ppt2pdf import Arguments

    return Arguments


def pw():
    from .pw import Arguments

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


def openrouter():
    from .openrouter import Arguments

    return Arguments


class Arguments(SubcommandArguments):
    aot = SubcommandSpec(name="aot", argument_class_factory=aot)
    any2md = SubcommandSpec(name="any2md", argument_class_factory=any2md)
    pdf2md = SubcommandSpec(name="pdf2md", argument_class_factory=pdf2md)
    pdf2txt = SubcommandSpec(name="pdf2txt", argument_class_factory=pdf2txt)
    web2md = SubcommandSpec(name="web2md", argument_class_factory=web2md)
    ppt2pdf = SubcommandSpec(name="ppt2pdf", argument_class_factory=ppt2pdf)
    makeppt = SubcommandSpec(name="makeppt", argument_class_factory=makeppt)
    pw = SubcommandSpec(name="pw", argument_class_factory=pw)
    transcribe = SubcommandSpec(name="transcribe", argument_class_factory=transcribe)
    upstage = SubcommandSpec(name="upstage", argument_class_factory=upstage)
    openrouter = SubcommandSpec(name="openrouter", argument_class_factory=openrouter)


def main():
    Arguments().execute()


if __name__ == "__main__":
    main()
