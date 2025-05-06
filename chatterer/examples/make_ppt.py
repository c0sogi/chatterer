def resolve_import_path_and_get_logger():
    # ruff: noqa: E402
    import logging
    import sys

    if __name__ == "__main__" and "." not in sys.path:
        sys.path.append(".")

    logger = logging.getLogger(__name__)
    return logger


logger = resolve_import_path_and_get_logger()
import re
import sys
from pathlib import Path
from typing import NotRequired, TypedDict

from spargear import BaseArguments

from chatterer import BaseMessage, Chatterer, HumanMessage, SystemMessage

# --- Default Prompts ---

DEFAULT_ROLE_PROMPT = """\
# Prompt Content

## Role Prompt: AI for Generating Presentation Slides and Scripts

You are a professional AI assistant that generates presentation slides and corresponding speech scripts based on user-provided content. Your primary task is to create a visually appealing and highly informative single presentation slide using HTML/CSS, along with a natural and well-structured speech script to accompany the slide. You must pay close attention to avoid layout breakage due to unnecessary whitespace or line breaks in text content or code blocks.

[Core Features]

1.  **Slide Generation (HTML/CSS):**
    *   Analyze the user's input, extract key information, and structure it logically.
    *   **Text Normalization:** Before inserting text into HTML elements, remove unintended line breaks and excessive spacing caused by OCR or formatting errors. Ensure that sentences and list items flow naturally. (e.g., "Easy gram\\nmar:" → "Easy grammar:")
    *   Design the slide based on a 16:9 aspect ratio (1280x720 pixels). Use the `.slide` class to explicitly define this size.
    *   Use Tailwind CSS utilities (via CDN: `<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">`)—such as Grid, Flexbox, Padding, and Margin—to structure the layout. Prevent layout issues caused by long text lines or unwanted white space.
    *   Enhance visuals with Font Awesome icons. Include the CDN in the `<head>` section (`<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css">`) and choose relevant icons appropriately.
    *   Use Korean Google Fonts: `<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap" rel="stylesheet">`
    *   **Code Block Handling:**
        *   Use `<pre><code>` tags for code samples. (e.g., `<div class="code-block bg-gray-800 text-white p-4 rounded mt-2"><pre><code>...code...</code></pre></div>`)
        *   Ensure that code lines inside `<code>` tags start without leading spaces to avoid rendering issues caused by HTML indentation.
        *   Optionally use `<span>` tags and classes like `comment`, `keyword`, or `string` for syntax highlighting.
    *   Carefully choose color, spacing, and typography to create a clean, professional layout.
    *   The generated HTML code must be fully self-contained, including all style information (Tailwind classes and optionally custom CSS via `<style>`) and CDN links. It should be enclosed in ```html ... ``` markdown blocks.

2.  **Speech Script Generation:**
    *   Write a clear and effective speech script to accompany the generated slide.
    *   Use a professional yet natural tone to ensure easy understanding for the audience.
    *   Explain each component of the slide (title, main points, code samples, etc.) in order, and include transitional or explanatory remarks where appropriate.
    *   Keep the speech concise but informative, covering all key points.
    *   Present the script clearly, outside of the HTML block.

[Workflow]

1.  Receive the user's input (topic or content for the presentation slide).
2.  Identify the key message, normalize the text, and structure it to fit into a single slide.
3.  Design and generate a 1280x720 slide using Tailwind CSS and Font Awesome. Pay special attention to code block formatting. Enclose the *complete* HTML for the slide within ```html ... ```.
4.  Write a natural speech script based on the slide's content. Place the script *after* the HTML block.
5.  Clearly separate the HTML code (within ```html ... ```) and the speech script.

[Guidelines]

*   Always aim to generate a single, self-contained slide.
*   Maintain high quality and structure similar to provided examples, but optimize the layout and content according to the input.
*   Pay close attention to whitespace and line break handling in both source HTML and rendered output, especially for plain text and code blocks.
*   Always use CDN links for external libraries (Tailwind, Font Awesome, Google Fonts).
*   Write the code with readability and reusability in mind.
*   The speech script must be tightly aligned with the slide content.
*   **Output Format:** First provide the complete HTML code block: ```html <!DOCTYPE html>...</html> ```. Immediately following the HTML block, provide the speech script as plain text.

Incorrect Code Block HTML Example (To Avoid):

```html
<!-- This renders unintended indentation due to <pre> preserving HTML source indentation -->
<div class="code-block">
    <pre><code>
        score = 85
        if score >= 60:
            print("Passed!")
        else:
            print("Failed.")
    </code></pre>
</div>
````

Correct Code Block HTML Example (Recommended):

```html
<div class="code-block bg-gray-800 text-white p-4 rounded">
<pre><code>score = 85
if score >= 60:
    print("Passed!") # Keep indentation within code block only
else:
    print("Failed.")</code></pre>
</div>
```"""

DEFAULT_JOB_PROMPT = """\
Objective: Analyze the content of the provided research material and create a detailed presentation slide plan in Markdown format. Save this plan to a file named 'plan.md'.

Detailed Guidelines:

1.  **Content Analysis:** Thoroughly analyze the provided research content to identify the core topic, main arguments, supporting data, conclusions, and other critical points.
2.  **Slide Structure:** Based on the analysis, organize the overall slide structure with a logical flow (e.g., Introduction – Main Body – Conclusion, or Problem – Solution). Aim for a reasonable number of slides to cover the material effectively without being overwhelming.
3.  **Slide-by-Slide Planning:** For each slide, define the content in detail. Each slide plan must include the following elements:
    *   **Slide Number:** Use the format `# Slide N`.
    *   **Topic:** Clearly state the central theme of the slide. (`Topic: ...`)
    *   **Summary:** Provide a concise summary of the key message, main content, relevant data, and possible visual ideas (like charts, key points, code snippets if applicable) to be included in the slide. (`Summary: ...`)
4.  **Output:** Generate *only* the Markdown content for the slide plan, following the format example below. Do not include any other explanatory text before or after the plan.

Output Format Example:

```markdown
# Slide 1

Topic: Background and Rationale
Summary: Explain why the [research topic] is important and what current issues it addresses. Briefly reference relevant statistics or previous studies to highlight the need for this research. Consider using a compelling opening statistic or image.

# Slide 2

Topic: Research Objectives and Questions
Summary: Clearly state the specific objectives of the research (e.g., To investigate X, To develop Y). Present 1–2 concise research questions that the study aims to answer. Use bullet points for clarity.

# Slide 3

Topic: Methodology
Summary: Describe the research methods used (e.g., surveys, literature review, experiments, data analysis techniques). Summarize the data collection process and key parameters. Maybe include a simple flowchart icon.

... (and so on for all necessary slides) ...

# Slide N

Topic: Conclusion and Recommendations
Summary: Summarize the key research findings and present the main conclusions clearly. Mention any significant limitations. Suggest directions for future research or practical recommendations based on the findings. End with a clear take-away message.
```

--- START OF RESEARCH MATERIAL ---
{research_content}
--- END OF RESEARCH MATERIAL ---

Now, generate the slide plan based *only* on the provided research material.
"""

DEFAULT_ORGANIZATION_PROMPT = """\
Objective: Create a final `presentation.html` file using impress.js (loaded via CDN) to structure the provided individual HTML slide contents into a cohesive presentation.

Guidelines:

1.  **Structure:** Use the standard impress.js HTML structure.
    *   Include the impress.js library and its default CSS via CDN in the `<head>`.
    *   Each slide's HTML content should be placed inside a `<div class="step">` element within the `<div id="impress">` container.
    *   Assign appropriate `data-x`, `data-y`, `data-scale`, `data-rotate`, etc., attributes to the `step` divs to create a logical flow and visual transitions between slides. A simple linear flow (increasing `data-x` for each step) is acceptable, but feel free to add minor variations if it enhances the presentation flow.
2.  **Content Integration:** Embed the *full* HTML content provided for each slide within its corresponding `<div class="step">`. Ensure the slide content (including its own `<head>` elements like Tailwind CSS links if they were generated per-slide) is correctly placed *inside* the `step` div. It might be better to consolidate CSS/Font links into the main HTML `<head>`. Let's aim to consolidate the CSS/Font links in the main `<head>` and put only the `<body>` content of each slide inside the `<div class="step">`.
3.  **impress.js Initialization:** Include the `impress().init();` script at the end of the `<body>`.
4.  **Output:** Generate *only* the complete HTML code for the `presentation.html` file. Do not include any other explanatory text.

--- START OF SLIDE HTML CONTENTS ---

{all_slides_html}

--- END OF SLIDE HTML CONTENTS ---

Now, generate the final `presentation.html` file using impress.js and the provided slide contents.
"""

# --- Argument Parsing ---


class MakePptArguments(BaseArguments):
    """
    Arguments for the presentation generation process.
    """

    # Input file paths
    research_file: str = "research.md"
    """Path to the input research file"""
    plan_file: str = "plan.md"
    """Path to the slide plan file"""
    output_file: str = "presentation.html"
    """Path to the output presentation file"""
    slides_dir: str = "slides"
    """Directory to save individual slide files"""

    # Prompt templates
    role_prompt: str = DEFAULT_ROLE_PROMPT
    """Role prompt for the AI assistant"""
    job_prompt: str = DEFAULT_JOB_PROMPT
    """Job prompt for content analysis and slide planning"""
    organization_prompt: str = DEFAULT_ORGANIZATION_PROMPT
    """Prompt for organizing slides into a presentation script"""

    # LLM Settings
    provider: str = (
        "openai:gpt-4.1"  # Example: "openai:gpt-4o", "anthropic:claude-3-haiku-20240307", "google:gemini-1.5-flash"
    )
    """Name of the language model to use (provider:model_name)"""

    # Other settings
    verbose: bool = True
    """Flag for verbose output during processing"""

    def run(self) -> None:
        # Create a dummy input file if it doesn't exist for testing
        if not Path(self.research_file).exists():
            print(f"Creating dummy input file: {self.research_file}")
            Path(self.research_file).write_text(
                """\
# Research Paper: The Future of AI Assistants

## Introduction
The field of Artificial Intelligence (AI) has seen exponential growth. AI assistants are becoming integrated into daily life. This research explores future trends.

## Current State
Current assistants (Siri, Alexa, Google Assistant) primarily handle simple commands, Q&A, and basic task automation. They rely heavily on predefined scripts and cloud connectivity. NLP has improved significantly, but true contextual understanding remains a challenge.

## Key Trends
1.  **Proactive Assistance:** Assistants will anticipate user needs.
2.  **Hyper-Personalization:** Tailoring responses and actions based on deep user understanding.
3.  **Multimodal Interaction:** Seamlessly integrating voice, text, vision, and gestures.
4.  **On-Device Processing:** Enhancing privacy and speed by reducing cloud dependency. Example: `model.run_locally()`
5.  **Emotional Intelligence:** Recognizing and responding appropriately to user emotions.

## Challenges
*   Data Privacy and Security
*   Algorithmic Bias
*   Computational Cost
*   Maintaining User Trust

## Conclusion
The future of AI assistants points towards highly personalized, proactive, and emotionally intelligent companions. Overcoming challenges related to privacy and bias is crucial for widespread adoption. Python code example:
```python
def greet(user):
    # Simple greeting
    print(f"Hello, {user}! How can I assist today?")

greet("Developer")
```
""",
                encoding="utf-8",
            )

        run_presentation_agent(self)


# --- Helper Functions ---


def parse_plan(plan_content: str) -> list[dict[str, str]]:
    """Parses the Markdown plan content into a list of slide dictionaries."""
    slides: list[dict[str, str]] = []
    # Regex to find slide blocks, topics, and summaries
    slide_pattern = re.compile(
        r"# Slide (?P<number>\d+)\s*\n+Topic:\s*(?P<topic>.*?)\s*\n+Summary:\s*(?P<summary>.*?)(?=\n+# Slide|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    matches = slide_pattern.finditer(plan_content)
    for match in matches:
        slide_data = match.groupdict()
        slides.append({
            "number": slide_data["number"].strip(),
            "topic": slide_data["topic"].strip(),
            "summary": slide_data["summary"].strip(),
        })
    return slides


def extract_html_script(response: str) -> tuple[str | None, str | None]:
    """Extracts HTML block and the following script from the AI response."""
    html_match = re.search(r"```html\s*(.*?)\s*```", response, re.DOTALL)
    if not html_match:
        # Maybe the AI didn't use ```html, try finding <!DOCTYPE html>...</html>
        html_match = re.search(r"<!DOCTYPE html>.*?</html>", response, re.DOTALL | re.IGNORECASE)
        if not html_match:
            return None, response  # Assume entire response might be script if no HTML found

    html_content = html_match.group(1) if len(html_match.groups()) > 0 else html_match.group(0)
    html_content = html_content.strip()

    # Script is assumed to be the text *after* the HTML block
    script_content = response[html_match.end() :].strip()

    return html_content, script_content if script_content else None


def extract_slide_body(html_content: str) -> str:
    """Extracts the content within the <body> tags of a slide's HTML."""
    body_match = re.search(r"<body.*?>(.*?)</body>", html_content, re.DOTALL | re.IGNORECASE)
    if body_match:
        return body_match.group(1).strip()
    else:
        # Fallback: If no body tag, return the whole content assuming it's body-only
        # Remove potential head/style tags if they exist outside body
        html_content = re.sub(r"<head>.*?</head>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r"<style>.*?</style>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
        return html_content.strip()


# --- Main Agent Loop ---


class GeneratedSlide(TypedDict):
    number: str
    html: str
    script: NotRequired[str]


def run_presentation_agent(args: MakePptArguments):
    """Executes the presentation generation agent loop."""

    if args.verbose:
        print(f"Initializing Presentation Agent with model: {args.provider}")
        print(f"Input file: {args.research_file}")
        print(f"Plan file: {args.plan_file}")
        print(f"Output file: {args.output_file}")
        print(f"Slides dir: {args.slides_dir}")

    # --- 1. Initialize Chatterer ---
    try:
        chatterer = Chatterer.from_provider(args.provider)
    except ValueError as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during model initialization: {e}")
        sys.exit(1)

    # --- 2. Read Input Research File ---
    input_path = Path(args.research_file)
    if not input_path.is_file():
        print(f"Error: Input file not found at '{args.research_file}'")
        sys.exit(1)

    try:
        research_content = input_path.read_text(encoding="utf-8")
        if args.verbose:
            print(f"Successfully read input file: {args.research_file}")
    except Exception as e:
        print(f"Error reading input file '{args.research_file}': {e}")
        sys.exit(1)

    # --- 3. Generate Plan ---
    plan_path = Path(args.plan_file)
    if args.verbose:
        print("\n--- Generating Presentation Plan ---")

    plan_prompt = args.job_prompt.format(research_content=research_content)
    messages = [HumanMessage(content=plan_prompt)]

    try:
        if args.verbose:
            print("Sending request to LLM for plan generation...")
        plan_content = chatterer(messages)
        if args.verbose:
            print("Received plan from LLM.")

        # Ensure the plan file's directory exists
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(plan_content, encoding="utf-8")
        if args.verbose:
            print(f"Presentation plan saved to: {args.plan_file}")
            # print(f"\nPlan Content:\n{plan_content[:500]}...\n") # Preview plan
    except Exception as e:
        print(f"Error during plan generation or saving: {e}")
        sys.exit(1)

    # --- 4. Parse Plan ---
    if args.verbose:
        print("\n--- Parsing Presentation Plan ---")
    try:
        slides_plan = parse_plan(plan_content)
        if not slides_plan:
            print("Error: Could not parse any slides from the generated plan. Check plan.md format.")
            sys.exit(1)
        if args.verbose:
            print(f"Successfully parsed {len(slides_plan)} slides from the plan.")
    except Exception as e:
        print(f"Error parsing plan file '{args.plan_file}': {e}")
        sys.exit(1)

    # --- 5. Generate Individual Slides ---
    if args.verbose:
        print("\n--- Generating Individual Slides & Scripts ---")

    slides_dir_path = Path(args.slides_dir)
    slides_dir_path.mkdir(parents=True, exist_ok=True)
    generated_slides_data: list[GeneratedSlide] = []  # Store {'html': ..., 'script': ...} for each slide

    system_message = SystemMessage(content=args.role_prompt)

    for i, slide_info in enumerate(slides_plan):
        slide_num = slide_info.get("number", str(i + 1))
        topic = slide_info.get("topic", "N/A")
        summary = slide_info.get("summary", "N/A")

        if args.verbose:
            print(f"\nGenerating Slide {slide_num}: {topic}")

        slide_gen_prompt = f"""\
Generate the HTML/CSS slide and the speech script for the following slide based on the plan:

Slide Number: {slide_num}
Topic: {topic}
Summary/Content Instructions:
{summary}

Remember to follow all instructions in the role prompt, especially regarding HTML structure (1280x720 .slide class, Tailwind, Font Awesome, Google Fonts via CDN, correct code block formatting) and output format (```html ... ``` block followed by the script).
"""
        messages: list[BaseMessage] = [system_message, HumanMessage(content=slide_gen_prompt)]

        try:
            if args.verbose:
                print("Sending request to LLM for slide generation...")
            response = chatterer(messages)
            if args.verbose:
                print("Received slide content from LLM.")

            html_content, script_content = extract_html_script(response)

            if not html_content:
                print(f"Warning: Could not extract HTML block for slide {slide_num}. Skipping.")
                continue

            slide_html_path = slides_dir_path / f"slide_{slide_num}.html"
            slide_script_path = slides_dir_path / f"slide_{slide_num}_script.txt"

            slide_html_path.write_text(html_content, encoding="utf-8")
            if args.verbose:
                print(f"Saved HTML for slide {slide_num} to: {slide_html_path}")

            if script_content:
                slide_script_path.write_text(script_content, encoding="utf-8")
                if args.verbose:
                    print(f"Saved script for slide {slide_num} to: {slide_script_path}")
            else:
                if args.verbose:
                    print(f"Warning: No script content found for slide {slide_num}.")

            generated_slides_data.append({"number": slide_num, "html": html_content, "script": script_content or ""})

        except Exception as e:
            print(f"Error generating slide {slide_num}: {e}")
            # Optionally continue to the next slide or exit
            # continue
            sys.exit(1)

    if not generated_slides_data:
        print("Error: No slides were successfully generated.")
        sys.exit(1)

    # --- 6. Organize Slides into Final Presentation ---
    if args.verbose:
        print("\n--- Organizing Slides into Final Presentation ---")

    output_path = Path(args.output_file)

    # Prepare the combined HTML content for the organization prompt
    # Extract only the body content of each slide
    all_slides_body_html = ""
    for slide_data in generated_slides_data:
        slide_body = extract_slide_body(slide_data["html"])
        all_slides_body_html += f"<!-- Slide {slide_data['number']} Content -->\n"
        all_slides_body_html += f"<div class='step' data-x='{int(slide_data['number']) * 1500}' data-y='0' data-scale='1'>\n"  # Simple linear layout
        all_slides_body_html += slide_body
        all_slides_body_html += "\n</div>\n\n"

    organization_formatted_prompt = args.organization_prompt.format(all_slides_html=all_slides_body_html)
    messages = [HumanMessage(content=organization_formatted_prompt)]

    try:
        if args.verbose:
            print("Sending request to LLM for final presentation generation...")
        final_presentation_html = chatterer(messages)

        # Often models add ```html markdown, remove it if present
        final_presentation_html = re.sub(r"^```html\s*", "", final_presentation_html, flags=re.IGNORECASE)
        final_presentation_html = re.sub(r"\s*```$", "", final_presentation_html)

        if args.verbose:
            print("Received final presentation HTML from LLM.")

        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(final_presentation_html, encoding="utf-8")
        if args.verbose:
            print(f"Final presentation saved to: {args.output_file}")

    except Exception as e:
        print(f"Error during final presentation generation or saving: {e}")
        sys.exit(1)

    print("\n--- Presentation Generation Complete! ---")


if __name__ == "__main__":
    MakePptArguments().run()
