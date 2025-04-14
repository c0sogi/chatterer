from pathlib import Path

from chatterer import Chatterer, PdfToMarkdown
from tqdm import tqdm

files = list(Path(r"C:\Users\cosogi\Desktop\로봇공학개론").rglob("*.pdf"))
files = [file for file in files if "2주차. Spatial description" not in file.parts]


chatterer = Chatterer.google("gemini-2.5-pro-preview-03-25")
converter = PdfToMarkdown(chatterer=chatterer)
for input in tqdm(files, desc="Converting"):
    output = input.with_suffix(".md")
    if output.is_file():
        print(f"Skipping {input} as .md already exists")
        continue
    print(f"Converting {input} to markdown ...")
    result = converter.convert(input.as_posix())
    output.write_text(result, encoding="utf-8")
    print(f"Saved to {output}")
