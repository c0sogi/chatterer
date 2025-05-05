from examples.pdf_to_markdown import PdfToMarkdownArgs, main

values = [
    r"c:\Users\cosogi\Downloads\1-s2.0-S0951832023000960-main.pdf",
    r"c:\Users\cosogi\Downloads\1-s2.0-S0951832023007731-main.pdf",
    r"c:\Users\cosogi\Downloads\1-s2.0-S1110016823011572-main.pdf",
    r"c:\Users\cosogi\Downloads\1-s2.0-S2352864823000354-main.pdf",
    r"c:\Users\cosogi\Downloads\1-s2.0-S0951832022006834-main.pdf",
]

for value in values:
    main(PdfToMarkdownArgs([value, "--chatterer", "google:gemini-2.5-flash-preview-04-17"]))
