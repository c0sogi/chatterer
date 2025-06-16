from __future__ import annotations

import asyncio
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

    This class supports both sequential and parallel processing:
    - Sequential processing preserves strict page continuity using previous page context
    - Parallel processing enables faster conversion for large documents by using
      previous page image and text for context instead of generated markdown
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
    """Number of lines from the end of the previous page's Markdown to use as context (sequential mode only)."""

    def _get_context_tail(self, markdown_text: Optional[str]) -> Optional[str]:
        """Extracts the last N lines from the given markdown text."""
        if not markdown_text or self.context_tail_lines <= 0:
            return None
        lines = markdown_text.strip().splitlines()
        if not lines:
            return None
        tail_lines = lines[-self.context_tail_lines :]
        return "\n".join(tail_lines)

    def _format_prompt_content_sequential(
        self,
        page_text: str,
        page_image_b64: Base64Image,
        previous_markdown_context_tail: Optional[str] = None,
        page_number: int = 0,
        total_pages: int = 1,
    ) -> HumanMessage:
        """
        Formats the content for sequential processing using previous page's markdown context.
        """
        instruction = f"""You are an expert PDF to Markdown converter. Convert Page {page_number + 1} of {total_pages} into accurate, well-formatted Markdown.

**Input provided:**
1. **Raw Text**: Extracted text from the PDF page (may contain OCR errors)
2. **Page Image**: Visual rendering of the page showing actual layout
3. **Previous Context**: End portion of the previous page's generated Markdown (if available)

**Conversion Rules:**
• **Text Structure**: Use the image to understand the actual layout and fix any OCR errors in the raw text
• **Headings**: Use appropriate heading levels (# ## ### etc.) based on visual hierarchy
• **Lists**: Convert to proper Markdown lists (- or 1. 2. 3.) maintaining structure
• **Tables**: Convert to Markdown table format using | pipes |
• **Images/Diagrams**: Describe significant visual elements as: `<details><summary>Figure: Brief title</summary>Detailed description based on what you see in the image</details>`
• **Code/Formulas**: Use ``` code blocks ``` or LaTeX $$ math $$ as appropriate
• **Continuity**: If previous context shows incomplete content (mid-sentence, list, table), seamlessly continue from that point
• **NO REPETITION**: Never repeat content from the previous context - only generate new content for this page

**Raw Text:**
```
{page_text if page_text else "No text extracted from this page."}
```

**Page Image:** (attached)
"""

        if previous_markdown_context_tail:
            instruction += f"""
**Previous Page Context (DO NOT REPEAT):**
```markdown
... (previous page ended with) ...
{previous_markdown_context_tail}
```

Continue seamlessly from the above context if the current page content flows from it.
"""
        else:
            instruction += "\n**Note:** This is the first page or start of a new section."

        instruction += "\n\n**Output only the Markdown content for the current page. Ensure proper formatting and NO repetition of previous content.**"

        return HumanMessage(content=[instruction, page_image_b64.data_uri_content_dict])

    def _format_prompt_content_parallel(
        self,
        page_text: str,
        page_image_b64: Base64Image,
        previous_page_text: Optional[str] = None,
        previous_page_image_b64: Optional[Base64Image] = None,
        page_number: int = 0,
        total_pages: int = 1,
    ) -> HumanMessage:
        """
        Formats the content for parallel processing using previous page's raw data.
        """
        instruction = f"""You are an expert PDF to Markdown converter. Convert Page {page_number + 1} of {total_pages} into accurate, well-formatted Markdown.

**Task**: Convert the current page to Markdown while maintaining proper continuity with the previous page.

**Current Page Data:**
- **Raw Text**: Extracted text (may have OCR errors - use image to verify)
- **Page Image**: Visual rendering showing actual layout

**Previous Page Data** (for context only):
- **Previous Raw Text**: Text from the previous page
- **Previous Page Image**: Visual of the previous page

**Conversion Instructions:**
1. **Primary Focus**: Convert the CURRENT page content accurately
2. **Continuity Check**: 
   - Examine if the current page continues content from the previous page (sentences, paragraphs, lists, tables)
   - If yes, start your Markdown naturally continuing that content
   - If no, start fresh with proper heading/structure
3. **Format Rules**:
   - Use image to fix OCR errors and understand layout
   - Convert headings to # ## ### based on visual hierarchy
   - Convert lists to proper Markdown (- or 1. 2. 3.)
   - Convert tables to | pipe | format
   - Describe significant images/charts as: `<details><summary>Figure: Title</summary>Description</details>`
   - Use ``` for code blocks and $$ for math formulas

**Current Page Raw Text:**
```
{page_text if page_text else "No text extracted from this page."}
```

**Current Page Image:** (see first attached image)
"""

        content: list[str | dict[str, object]] = [instruction, page_image_b64.data_uri_content_dict]

        if previous_page_text is not None and previous_page_image_b64 is not None:
            instruction += f"""

**Previous Page Raw Text (for context):**
```
{previous_page_text if previous_page_text else "No text from previous page."}
```

**Previous Page Image:** (see second attached image)
"""
            content.append(previous_page_image_b64.data_uri_content_dict)
        else:
            instruction += "\n**Note:** This is the first page - no previous context available."

        instruction += (
            "\n\n**Generate ONLY the Markdown for the current page. Ensure proper continuity and formatting.**"
        )
        content[0] = instruction

        return HumanMessage(content=content)

    def convert(
        self,
        pdf_input: "Document | PathOrReadable",
        page_indices: Optional[PageIndexType] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        mode: Literal["sequential", "parallel"] = "sequential",
    ) -> str:
        """
        Converts a PDF document to Markdown synchronously.

        Args:
            pdf_input: Path to PDF file or pymupdf.Document object
            page_indices: Specific page indices to convert (0-based). If None, converts all pages
            progress_callback: Optional callback function called with (current_page, total_pages)
            mode: "sequential" for strict continuity or "parallel" for independent page processing

        Returns:
            Concatenated Markdown string for all processed pages
        """
        if mode == "sequential":
            return self._convert_sequential(pdf_input, page_indices, progress_callback)
        else:
            return self._convert_parallel_sync(pdf_input, page_indices, progress_callback)

    async def aconvert(
        self,
        pdf_input: "Document | PathOrReadable",
        page_indices: Optional[PageIndexType] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        max_concurrent: int = 5,
    ) -> str:
        """
        Converts a PDF document to Markdown asynchronously with parallel processing.

        Args:
            pdf_input: Path to PDF file or pymupdf.Document object
            page_indices: Specific page indices to convert (0-based). If None, converts all pages
            progress_callback: Optional callback function called with (current_page, total_pages)
            max_concurrent: Maximum number of concurrent LLM requests

        Returns:
            Concatenated Markdown string for all processed pages
        """
        with open_pdf(pdf_input) as doc:
            target_page_indices = list(
                _get_page_indices(page_indices=page_indices, max_doc_pages=len(doc), is_input_zero_based=True)
            )
            total_pages_to_process = len(target_page_indices)

            if total_pages_to_process == 0:
                logger.warning("No pages selected for processing.")
                return ""

            logger.info(f"Starting parallel Markdown conversion for {total_pages_to_process} pages...")

            # Pre-process all pages
            page_text_dict = extract_text_from_pdf(doc, target_page_indices)
            page_image_dict = render_pdf_as_image(
                doc,
                page_indices=target_page_indices,
                zoom=self.image_zoom,
                output=self.image_format,
                jpg_quality=self.image_jpg_quality,
            )

            # Process pages in parallel with semaphore for concurrency control
            semaphore = asyncio.Semaphore(max_concurrent)

            async def process_page(i: int, page_idx: int) -> tuple[int, str]:
                async with semaphore:
                    logger.info(f"Processing page {i + 1}/{total_pages_to_process} (Index: {page_idx})...")

                    try:
                        # Get previous page data for context
                        prev_page_idx = target_page_indices[i - 1] if i > 0 else None
                        previous_page_text = page_text_dict.get(prev_page_idx) if prev_page_idx is not None else None
                        previous_page_image_b64 = None
                        if prev_page_idx is not None:
                            previous_page_image_b64 = Base64Image.from_bytes(
                                page_image_dict[prev_page_idx], ext=self.image_format
                            )

                        message = self._format_prompt_content_parallel(
                            page_text=page_text_dict.get(page_idx, ""),
                            page_image_b64=Base64Image.from_bytes(page_image_dict[page_idx], ext=self.image_format),
                            previous_page_text=previous_page_text,
                            previous_page_image_b64=previous_page_image_b64,
                            page_number=page_idx,
                            total_pages=len(doc),
                        )

                        response = await self.chatterer.agenerate([message])

                        # Extract markdown
                        markdowns = [match.group(1).strip() for match in MARKDOWN_PATTERN.finditer(response)]
                        if markdowns:
                            current_page_markdown = "\n".join(markdowns)
                        else:
                            current_page_markdown = response.strip()
                            if current_page_markdown.startswith("```") and current_page_markdown.endswith("```"):
                                current_page_markdown = current_page_markdown[3:-3].strip()

                        logger.debug(f"Completed processing page {i + 1}/{total_pages_to_process}")

                        # Call progress callback if provided
                        if progress_callback:
                            try:
                                progress_callback(i + 1, total_pages_to_process)
                            except Exception as cb_err:
                                logger.warning(f"Progress callback failed: {cb_err}")

                        return (i, current_page_markdown)

                    except Exception as e:
                        logger.error(f"Failed to process page index {page_idx}: {e}", exc_info=True)
                        return (i, f"<!-- Error processing page {page_idx + 1}: {str(e)} -->")

                        # Execute all page processing tasks

            tasks = [process_page(i, page_idx) for i, page_idx in enumerate(target_page_indices)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Sort results by original page order and extract markdown
            markdown_results = [""] * total_pages_to_process
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed with exception: {result}")
                    continue
                if isinstance(result, tuple) and len(result) == 2:
                    page_order, markdown = result
                    markdown_results[page_order] = markdown
                else:
                    logger.error(f"Unexpected result format: {result}")

            return "\n\n".join(markdown_results).strip()

    def _convert_sequential(
        self,
        pdf_input: "Document | PathOrReadable",
        page_indices: Optional[PageIndexType] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """Sequential conversion maintaining strict page continuity."""
        with open_pdf(pdf_input) as doc:
            target_page_indices = list(
                _get_page_indices(page_indices=page_indices, max_doc_pages=len(doc), is_input_zero_based=True)
            )
            total_pages_to_process = len(target_page_indices)
            if total_pages_to_process == 0:
                logger.warning("No pages selected for processing.")
                return ""

            full_markdown_output: List[str] = []
            previous_page_markdown: Optional[str] = None

            # Pre-process all pages
            logger.info("Extracting text and rendering images for selected pages...")
            page_text_dict = extract_text_from_pdf(doc, target_page_indices)
            page_image_dict = render_pdf_as_image(
                doc,
                page_indices=target_page_indices,
                zoom=self.image_zoom,
                output=self.image_format,
                jpg_quality=self.image_jpg_quality,
            )
            logger.info(f"Starting sequential Markdown conversion for {total_pages_to_process} pages...")

            for i, page_idx in enumerate(target_page_indices):
                logger.info(f"Processing page {i + 1}/{total_pages_to_process} (Index: {page_idx})...")
                try:
                    context_tail = self._get_context_tail(previous_page_markdown)

                    message = self._format_prompt_content_sequential(
                        page_text=page_text_dict.get(page_idx, ""),
                        page_image_b64=Base64Image.from_bytes(page_image_dict[page_idx], ext=self.image_format),
                        previous_markdown_context_tail=context_tail,
                        page_number=page_idx,
                        total_pages=len(doc),
                    )

                    response = self.chatterer.generate([message])

                    # Extract markdown
                    markdowns = [match.group(1).strip() for match in MARKDOWN_PATTERN.finditer(response)]
                    if markdowns:
                        current_page_markdown = "\n".join(markdowns)
                    else:
                        current_page_markdown = response.strip()
                        if current_page_markdown.startswith("```") and current_page_markdown.endswith("```"):
                            current_page_markdown = current_page_markdown[3:-3].strip()

                    full_markdown_output.append(current_page_markdown)
                    previous_page_markdown = current_page_markdown

                except Exception as e:
                    logger.error(f"Failed to process page index {page_idx}: {e}", exc_info=True)
                    continue

                # Progress callback
                if progress_callback:
                    try:
                        progress_callback(i + 1, total_pages_to_process)
                    except Exception as cb_err:
                        logger.warning(f"Progress callback failed: {cb_err}")

            return "\n\n".join(full_markdown_output).strip()

    def _convert_parallel_sync(
        self,
        pdf_input: "Document | PathOrReadable",
        page_indices: Optional[PageIndexType] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """Synchronous parallel-style conversion (processes independently but sequentially)."""
        with open_pdf(pdf_input) as doc:
            target_page_indices = list(
                _get_page_indices(page_indices=page_indices, max_doc_pages=len(doc), is_input_zero_based=True)
            )
            total_pages_to_process = len(target_page_indices)
            if total_pages_to_process == 0:
                logger.warning("No pages selected for processing.")
                return ""

            logger.info(f"Starting parallel-style Markdown conversion for {total_pages_to_process} pages...")

            # Pre-process all pages
            page_text_dict = extract_text_from_pdf(doc, target_page_indices)
            page_image_dict = render_pdf_as_image(
                doc,
                page_indices=target_page_indices,
                zoom=self.image_zoom,
                output=self.image_format,
                jpg_quality=self.image_jpg_quality,
            )

            full_markdown_output: List[str] = []

            for i, page_idx in enumerate(target_page_indices):
                logger.info(f"Processing page {i + 1}/{total_pages_to_process} (Index: {page_idx})...")

                try:
                    # Get previous page data for context
                    prev_page_idx = target_page_indices[i - 1] if i > 0 else None
                    previous_page_text = page_text_dict.get(prev_page_idx) if prev_page_idx is not None else None
                    previous_page_image_b64 = None
                    if prev_page_idx is not None:
                        previous_page_image_b64 = Base64Image.from_bytes(
                            page_image_dict[prev_page_idx], ext=self.image_format
                        )

                    message = self._format_prompt_content_parallel(
                        page_text=page_text_dict.get(page_idx, ""),
                        page_image_b64=Base64Image.from_bytes(page_image_dict[page_idx], ext=self.image_format),
                        previous_page_text=previous_page_text,
                        previous_page_image_b64=previous_page_image_b64,
                        page_number=page_idx,
                        total_pages=len(doc),
                    )

                    response = self.chatterer.generate([message])

                    # Extract markdown
                    markdowns = [match.group(1).strip() for match in MARKDOWN_PATTERN.finditer(response)]
                    if markdowns:
                        current_page_markdown = "\n".join(markdowns)
                    else:
                        current_page_markdown = response.strip()
                        if current_page_markdown.startswith("```") and current_page_markdown.endswith("```"):
                            current_page_markdown = current_page_markdown[3:-3].strip()

                    full_markdown_output.append(current_page_markdown)

                except Exception as e:
                    logger.error(f"Failed to process page index {page_idx}: {e}", exc_info=True)
                    continue

                # Progress callback
                if progress_callback:
                    try:
                        progress_callback(i + 1, total_pages_to_process)
                    except Exception as cb_err:
                        logger.warning(f"Progress callback failed: {cb_err}")

            return "\n\n".join(full_markdown_output).strip()


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
