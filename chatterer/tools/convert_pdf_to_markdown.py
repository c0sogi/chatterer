from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from types import EllipsisType
from typing import TYPE_CHECKING, Callable, Iterable, List, Literal, Optional

from ..language_model import Chatterer, HumanMessage
from ..utils.base64_image import Base64Image
from ..utils.bytesio import PathOrReadable, read_bytes_stream

if TYPE_CHECKING:
    from pymupdf import Document  # pyright: ignore[reportMissingTypeStubs]

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
MARKDOWN_PATTERN: re.Pattern[str] = re.compile(r"```(?:markdown\s*\n)?(.*?)```", re.DOTALL)
PageIndexType = Iterable[int | tuple[int | EllipsisType, int | EllipsisType]] | int | str


@dataclass
class PdfToMarkdown:
    """
    Converts PDF documents to Markdown using a multimodal LLM (Chatterer).
    Processes PDFs page by page, providing the LLM with both the extracted raw
    text and a rendered image of the page to handle complex layouts. It maintains
    context between pages by feeding the *tail end* of the previously generated
    Markdown back into the prompt for the next page to ensure smooth transitions.
    """

    chatterer: Chatterer
    """An instance of the Chatterer class configured with a vision-capable model."""
    image_zoom: float = 2.0
    """Zoom factor for rendering PDF pages as images (higher zoom = higher resolution)."""
    image_format: Literal["jpg", "jpeg", "png"] = "png"
    """The format for the rendered image ('png', 'jpeg', 'jpg'.)."""
    image_jpg_quality: int = 95
    """Quality for JPEG images (if used)."""
    context_tail_lines: int = 10
    """Number of lines from the end of the previous page's Markdown to use as context."""
    # max_context_tokens: Optional[int] = None # This can be added later if needed

    def _get_context_tail(self, markdown_text: Optional[str]) -> Optional[str]:
        """Extracts the last N lines from the given markdown text."""
        if not markdown_text or self.context_tail_lines <= 0:
            return None
        lines = markdown_text.strip().splitlines()
        if not lines:
            return None
        # Get the last N lines, or fewer if the text is shorter
        tail_lines = lines[-self.context_tail_lines :]
        return "\n".join(tail_lines)

    def _format_prompt_content(
        self,
        page_text: str,
        page_image_b64: Base64Image,
        previous_markdown_context_tail: Optional[str] = None,  # Renamed for clarity
        page_number: int = 0,  # For context, 0-indexed
        total_pages: int = 1,
    ) -> HumanMessage:
        """
        Formats the content list for the HumanMessage input to the LLM.
        Uses only the tail end of the previous page's markdown for context.
        """
        # Construct the main instruction prompt
        instruction = f"""You are an expert PDF to Markdown converter. Your task is to convert the content of the provided PDF page (Page {page_number + 1} of {total_pages}) into accurate and well-formatted Markdown. You are given:
1.  The raw text extracted from the page ([Raw Text]).
2.  A rendered image of the page ([Rendered Image]) showing its visual layout.
3.  (Optional) The *ending portion* of the Markdown generated from the previous page ([End of Previous Page Markdown]) for context continuity.

**Conversion Requirements:**
*   **Text:** Reconstruct paragraphs, headings, lists, etc., naturally based on the visual layout. Correct OCR/formatting issues from [Raw Text] using the image. Minimize unnecessary whitespace.
*   **Tables:** Convert tables accurately into Markdown table format (`| ... |`). Use image for text if [Raw Text] is garbled.
*   **Images/Diagrams:** Describe significant visual elements (charts, graphs) within `<details>` tags. Example: `<details><summary>Figure 1: Description</summary>Detailed textual description from the image.</details>`. Ignore simple decorative images. Do **not** use `![alt](...)`.
*   **Layout:** Respect columns, code blocks (``` ```), footnotes, etc., using standard Markdown.
*   **Continuity (Crucial):**
    *   Examine the [End of Previous Page Markdown] if provided.
    *   If the current page's content *continues* a sentence, paragraph, list, or code block from the previous page, ensure your generated Markdown for *this page* starts seamlessly from that continuation point.
    *   For example, if the previous page ended mid-sentence, the Markdown for *this page* should begin with the rest of that sentence.
    *   **Do NOT repeat the content already present in [End of Previous Page Markdown] in your output.**
    *   If the current page starts a new section (e.g., with a heading), begin the Markdown output fresh, ignoring the previous context tail unless necessary for list numbering, etc.

**Input Data:**
[Raw Text]
```
{page_text if page_text else "No text extracted from this page."}
```
[Rendered Image]
(See attached image)
"""
        if previous_markdown_context_tail:
            instruction += f"""[End of Previous Page Markdown]
```markdown
... (content from previous page ends with) ...
{previous_markdown_context_tail}
```
**Task:** Generate the Markdown for the *current* page (Page {page_number + 1}), ensuring it correctly continues from or follows the [End of Previous Page Markdown]. Start the output *only* with the content belonging to the current page."""
        else:
            instruction += "**Task:** Generate the Markdown for the *current* page (Page {page_number + 1}). This is the first page being processed in this batch."

        instruction += "\n\n**Output only the Markdown content for the current page.** Ensure your output starts correctly based on the continuity rules."

        # Structure for multimodal input
        return HumanMessage(content=[instruction, page_image_b64.data_uri_content])

    def convert(
        self,
        pdf_input: "Document | PathOrReadable",
        page_indices: Optional[PageIndexType] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """
        Converts a PDF document (or specific pages) to Markdown synchronously.
        Args:
            pdf_input: Path to the PDF file or a pymupdf.Document object.
            page_indices: Specific 0-based page indices to convert. If None, converts all pages.
                          Can be a single int or an iterable of ints.
            progress_callback: An optional function to call with (current_page_index, total_pages_to_process)
                               after each page is processed.
        Returns:
            A single string containing the concatenated Markdown output for the processed pages.
        """
        with open_pdf(pdf_input) as doc:
            target_page_indices = list(
                _get_page_indices(page_indices=page_indices, max_doc_pages=len(doc), is_input_zero_based=True)
            )
            total_pages_to_process = len(target_page_indices)
            if total_pages_to_process == 0:
                logger.warning("No pages selected for processing.")
                return ""

            full_markdown_output: List[str] = []
            # --- Context Tracking ---
            previous_page_markdown: Optional[str] = None  # Store the full markdown of the previous page

            # Pre-process all pages (optional optimization)
            logger.info("Extracting text and rendering images for selected pages...")
            page_text_dict = extract_text_from_pdf(doc, target_page_indices)
            page_image_dict = render_pdf_as_image(
                doc,
                page_indices=target_page_indices,
                zoom=self.image_zoom,
                output=self.image_format,
                jpg_quality=self.image_jpg_quality,
            )
            logger.info(f"Starting Markdown conversion for {total_pages_to_process} pages...")

            page_idx: int = target_page_indices.pop(0)  # Get the first page index
            i: int = 1
            while True:
                logger.info(f"Processing page {i}/{total_pages_to_process} (Index: {page_idx})...")
                try:
                    # --- Get Context Tail ---
                    context_tail = self._get_context_tail(previous_page_markdown)

                    message = self._format_prompt_content(
                        page_text=page_text_dict.get(page_idx, ""),  # Use .get for safety
                        page_image_b64=Base64Image.from_bytes(page_image_dict[page_idx], ext=self.image_format),
                        previous_markdown_context_tail=context_tail,  # Pass only the tail
                        page_number=page_idx,
                        total_pages=len(doc),
                    )
                    logger.debug(f"Sending request to LLM for page index {page_idx}...")

                    response = self.chatterer([message])
                    # Extract markdown, handling potential lack of backticks
                    markdowns: list[str] = [match.group(1).strip() for match in MARKDOWN_PATTERN.finditer(response)]
                    if markdowns:
                        current_page_markdown = "\n".join(markdowns)
                    else:
                        # Fallback: assume the whole response is markdown if no ```markdown blocks found
                        current_page_markdown = response.strip()
                        if current_page_markdown.startswith("```") and current_page_markdown.endswith("```"):
                            # Basic cleanup if it just missed the 'markdown' language tag
                            current_page_markdown = current_page_markdown[3:-3].strip()
                        elif "```" in current_page_markdown:
                            logger.warning(
                                f"Page {page_idx + 1}: Response contains '```' but not in expected format. Using raw response."
                            )

                    logger.debug(f"Received response from LLM for page index {page_idx}.")

                    # --- Store result and update context ---
                    full_markdown_output.append(current_page_markdown)
                    # Update the *full* previous markdown for the *next* iteration's tail calculation
                    previous_page_markdown = current_page_markdown

                except Exception as e:
                    logger.error(f"Failed to process page index {page_idx}: {e}", exc_info=True)
                    continue

                # Progress callback
                if progress_callback:
                    try:
                        progress_callback(i, total_pages_to_process)
                    except Exception as cb_err:
                        logger.warning(f"Progress callback failed: {cb_err}")

                if not target_page_indices:
                    break

                page_idx = target_page_indices.pop(0)  # Get the next page index
                i += 1  # Increment the page counter

        # Join with double newline, potentially adjust based on how well continuations work
        return "\n\n".join(full_markdown_output).strip()  # Add strip() to remove leading/trailing whitespace


def render_pdf_as_image(
    doc: "Document",
    zoom: float = 2.0,
    output: Literal["png", "pnm", "pgm", "ppm", "pbm", "pam", "tga", "tpic", "psd", "ps", "jpg", "jpeg"] = "png",
    jpg_quality: int = 100,
    page_indices: Iterable[int] | int | None = None,
) -> dict[int, bytes]:
    """
    Convert PDF pages to images in bytes.

    Args:
        doc (Document): The PDF document to convert.
        zoom (float): Zoom factor for the image resolution. Default is 2.0.
        output (str): Output format for the image. Default is 'png'.
        jpg_quality (int): Quality of JPEG images (1-100). Default is 100.
        page_indices (Iterable[int] | int | None): Specific pages to convert. If None, all pages are converted.
            If an int is provided, only that page is converted.

    Returns:
        dict[int, bytes]: A dictionary mapping page numbers to image bytes.
    """
    from pymupdf import Matrix  # pyright: ignore[reportMissingTypeStubs]
    from pymupdf.utils import get_pixmap  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

    images_bytes: dict[int, bytes] = {}
    matrix = Matrix(zoom, zoom)  # Control output resolution
    for page_idx in _get_page_indices(page_indices=page_indices, max_doc_pages=len(doc), is_input_zero_based=True):
        img_bytes = bytes(
            get_pixmap(
                page=doc[page_idx],
                matrix=matrix,
            ).tobytes(output=output, jpg_quality=jpg_quality)  # pyright: ignore[reportUnknownArgumentType]
        )
        images_bytes[page_idx] = img_bytes
    return images_bytes


def extract_text_from_pdf(doc: "Document", page_indices: Optional[PageIndexType] = None) -> dict[int, str]:
    """Convert a PDF file to plain text.

    Extracts text from each page of a PDF file and formats it with page markers.

    Args:
        doc (Document): The PDF document to convert.
        page_indices (Iterable[int] | int | None): Specific pages to convert. If None, all pages are converted.
            If an int is provided, only that page is converted.

    Returns:
        dict[int, str]: A dictionary mapping page numbers to text content.
    """
    return {
        page_idx: doc[page_idx].get_textpage().extractText().strip()  # pyright: ignore[reportUnknownMemberType]
        for page_idx in _get_page_indices(
            page_indices=page_indices,
            max_doc_pages=len(doc),
            is_input_zero_based=True,
        )
    }


@contextmanager
def open_pdf(pdf_input: PathOrReadable | Document):
    """Open a PDF document from a file path or use an existing Document object.

    Args:
        pdf_input (PathOrReadable | Document): The PDF file path or a pymupdf.Document object.

    Returns:
        tuple[Document, bool]: A tuple containing the opened Document object and a boolean indicating if it was opened internally.
    """
    import pymupdf  # pyright: ignore[reportMissingTypeStubs]

    should_close = True

    if isinstance(pdf_input, pymupdf.Document):
        should_close = False
        doc = pdf_input
    else:
        with read_bytes_stream(pdf_input) as stream:
            if stream is None:
                raise FileNotFoundError(pdf_input)
            doc = pymupdf.Document(stream=stream.read())
    yield doc
    if should_close:
        doc.close()


def _get_page_indices(
    page_indices: Optional[PageIndexType], max_doc_pages: int, is_input_zero_based: bool
) -> list[int]:
    """Helper function to handle page indices for PDF conversion."""

    def _to_zero_based_int(idx: int) -> int:
        """Convert a 1-based index to a 0-based index if necessary."""
        if is_input_zero_based:
            return idx
        else:
            if idx < 1 or idx > max_doc_pages:
                raise ValueError(f"Index {idx} is out of bounds for document with {max_doc_pages} pages (1-based).")
            return idx - 1

    if page_indices is None:
        return list(range(max_doc_pages))  # Convert all pages
    elif isinstance(page_indices, int):
        # Handle single integer input for page index
        return [_to_zero_based_int(page_indices)]
    elif isinstance(page_indices, str):
        # Handle string input for page indices
        return _interpret_index_string(
            index_str=page_indices, max_doc_pages=max_doc_pages, is_input_zero_based=is_input_zero_based
        )
    else:
        # Handle iterable input for page indices
        indices: set[int] = set()
        for idx in page_indices:
            if isinstance(idx, int):
                indices.add(_to_zero_based_int(idx))
            else:
                start, end = idx
                if isinstance(start, EllipsisType):
                    start = 0
                else:
                    start = _to_zero_based_int(start)

                if isinstance(end, EllipsisType):
                    end = max_doc_pages - 1
                else:
                    end = _to_zero_based_int(end)

                if start > end:
                    raise ValueError(
                        f"Invalid range: {start} - {end}. Start index must be less than or equal to end index."
                    )
                indices.update(range(start, end + 1))

        return sorted(indices)  # Return sorted list of indices


def _interpret_index_string(index_str: str, max_doc_pages: int, is_input_zero_based: bool) -> list[int]:
    """Interpret a string of comma-separated indices and ranges."""

    def _to_zero_based_int(idx_str: str) -> int:
        i = int(idx_str)
        if is_input_zero_based:
            if i < 0 or i >= max_doc_pages:
                raise ValueError(f"Index {i} is out of bounds for document with {max_doc_pages} pages.")
            return i
        else:
            if i < 1 or i > max_doc_pages:
                raise ValueError(f"Index {i} is out of bounds for document with {max_doc_pages} pages (1-based).")
            return i - 1  # Convert to zero-based index

    indices: set[int] = set()
    for part in index_str.split(","):
        part: str = part.strip()
        count_dash: int = part.count("-")
        if count_dash == 0:
            indices.add(_to_zero_based_int(part))
        elif count_dash == 1:
            idx_dash: int = part.index("-")
            start = part[:idx_dash].strip()
            end = part[idx_dash + 1 :].strip()
            if not start:
                start = _to_zero_based_int("0")  # Default to 0 if no start index is provided
            else:
                start = _to_zero_based_int(start)

            if not end:
                end = _to_zero_based_int(str(max_doc_pages - 1))  # Default to last page if no end index is provided
            else:
                end = _to_zero_based_int(end)

            if start > end:
                raise ValueError(
                    f"Invalid range: {start} - {end}. Start index must be less than or equal to end index."
                )
            indices.update(range(start, end + 1))
        else:
            raise ValueError(f"Invalid page index format: '{part}'. Expected format is '1,2,3' or '1-3'.")

    return sorted(indices)  # Return sorted list of indices, ensuring no duplicates
