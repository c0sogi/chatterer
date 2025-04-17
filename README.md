# Chatterer

**Simplified, Structured AI Assistant Framework**

`chatterer` is a Python library designed as a type-safe LangChain wrapper for interacting with various language models (OpenAI, Anthropic, Google Gemini, Ollama, etc.). It supports structured outputs via Pydantic models, plain text responses, asynchronous calls, image description, code execution, and an interactive shell.

The structured reasoning in `chatterer` is inspired by the [Atom-of-Thought](https://github.com/qixucen/atom) pipeline.

---

## Quick Install

```bash
pip install chatterer
```

---

## Quickstart Example

Generate text quickly using OpenAI. 
Messages can be input as plain strings or structured lists:

```python
from chatterer import Chatterer, HumanMessage, AIMessage, SystemMessage

# Initialize the Chatterer with `openai`, `anthropic`, `google`, or `ollama` models
chatterer: Chatterer = Chatterer.openai("gpt-4.1")

# Get direct response as str
response: str = chatterer("What is the meaning of life?")
# response = chatterer([{ "role": "user", "content": "What is the meaning of life?" }])
# response = chatterer([("user", "What is the meaning of life?")])
# response = chatterer([HumanMessage("What is the meaning of life?")])
print(response)
```

Image & text content can be sent as together:

```python
from chatterer import Base64Image, HumanMessage

# Load an image from a file or URL, resulting in a None or Base64Image object
image = Base64Image.from_url_or_path("example.jpg")
# image = Base64Image.from_url_or_path("https://example.com/image.jpg")
assert image is not None, "Failed to load image"

# Alternatively, load an image from bytes
# with open("example.jpg", "rb") as f:
#     image = Base64Image.from_bytes(f.read(), ext="jpeg")

message = HumanMessage(["Describe the image", image.data_uri_content])
response: str = chatterer([message])
print(response)
```

---

## Structured Output with Pydantic

Define a Pydantic model and get typed responses:

```python
from pydantic import BaseModel

class AnswerModel(BaseModel):
    question: str
    answer: str

# Call with response_model
response: AnswerModel = chatterer("What's the capital of France?", response_model=AnswerModel)
print(response.question, response.answer)
```

---

## Async Example

Use asynchronous generation for non-blocking operations:

```python
import asyncio

async def main():
    response = await chatterer.agenerate("Explain async in Python briefly.")
    print(response)

asyncio.run(main())
```

---

## Streaming Structured Outputs

Stream structured responses in real-time:

```python
from pydantic import BaseModel

class AnswerModel(BaseModel):
    text: str

chatterer = Chatterer.openai()
for chunk in chatterer.generate_pydantic_stream(AnswerModel, "Tell me a story"):
    print(chunk.text)
```

Asynchronous version:
```python
import asyncio

async def main():
    async for chunk in chatterer.agenerate_pydantic_stream(AnswerModel, "Tell me a story"):
        print(chunk.text)

asyncio.run(main())
```

---

## Image Description

Generate descriptions for images using the language model:

```python
description = chatterer.describe_image("https://example.com/image.jpg")
print(description)

# Customize the instruction
description = chatterer.describe_image("https://example.com/image.jpg", instruction="Describe the main objects in the image.")
```

An asynchronous version is also available:

```python
async def main():
    description = await chatterer.adescribe_image("https://example.com/image.jpg")
    print(description)

asyncio.run(main())
```

---

## Code Execution

Generate and execute Python code dynamically:

```python
result = chatterer.invoke_code_execution("Write a function to calculate factorial.")
print(result.code)
print(result.output)
```

An asynchronous version exists as well:

```python
async def main():
    result = await chatterer.ainvoke_code_execution("Write a function to calculate factorial.")
    print(result.output)

asyncio.run(main())
```

---

## Webpage to Markdown

Convert webpages to Markdown, optionally filtering content with the language model:

```python
from chatterer.tools.webpage_to_markdown import PlayWrightBot

with PlayWrightBot() as bot:
    # Basic conversion
    markdown = bot.url_to_md("https://example.com")
    print(markdown)

    # With LLM filtering and image descriptions
    filtered_md = bot.url_to_md_with_llm("https://example.com", describe_images=True)
    print(filtered_md)
```

Asynchronous version:
```python
import asyncio

async def main():
    async with PlayWrightBot() as bot:
        markdown = await bot.aurl_to_md_with_llm("https://example.com")
        print(markdown)

asyncio.run(main())
```

Extract specific elements:
```python
with PlayWrightBot() as bot:
    headings = bot.select_and_extract("https://example.com", "h2")
    print(headings)
```

---

## Citation Chunking

Chunk documents into semantic sections with citations:

```python
from chatterer import Chatterer
from chatterer.tools import citation_chunker

chatterer = Chatterer.openai()
document = "Long text about quantum computing..."
chunks = citation_chunker(document, chatterer, global_coverage_threshold=0.9)
for chunk in chunks:
    print(f"Subject: {chunk.name}")
    for source, matches in chunk.references.items():
        print(f"  Source: {source}, Matches: {matches}")
```

---

## Interactive Shell

Engage in a conversational AI session with code execution support:

```python
from chatterer import interactive_shell

interactive_shell()
```

This launches an interactive session where you can chat with the AI and execute code snippets. Type `quit` or `exit` to end the session.

---

## Atom-of-Thought Pipeline (AoT)

`AoTPipeline` provides structured reasoning inspired by the [Atom-of-Thought](https://github.com/qixucen/atom) approach. It decomposes complex questions recursively, generates answers, and combines them via an ensemble process.

### AoT Usage Example

```python
from chatterer import Chatterer
from chatterer.strategies import AoTStrategy, AoTPipeline

pipeline = AoTPipeline(chatterer=Chatterer.openai(), max_depth=2)
strategy = AoTStrategy(pipeline=pipeline)

question = "What would Newton discover if hit by an apple falling from 100 meters?"
answer = strategy.invoke(question)
print(answer)

# Generate and inspect reasoning graph
graph = strategy.get_reasoning_graph()
print(f"Graph: {len(graph.nodes)} nodes, {len(graph.relationships)} relationships")
```

**Note**: The AoT pipeline includes an optional feature to generate a reasoning graph, which can be stored in Neo4j for visualization and analysis. Install `neo4j_extension` and set up a Neo4j instance to use this feature:

```python
from neo4j_extension import Neo4jConnection
with Neo4jConnection() as conn:
    conn.upsert_graph(graph)
```

---

## Supported Models

Chatterer supports multiple language models, easily initialized as follows:

- **OpenAI**
- **Anthropic**
- **Google Gemini**
- **Ollama** (local models)

```python
openai_chatterer = Chatterer.openai("gpt-4o-mini")
anthropic_chatterer = Chatterer.anthropic("claude-3-7-sonnet-20250219")
gemini_chatterer = Chatterer.google("gemini-2.0-flash")
ollama_chatterer = Chatterer.ollama("deepseek-r1:1.5b")
```

---

## Advanced Features

- **Streaming Responses**: Use `generate_stream` or `agenerate_stream` for real-time output.
- **Streaming Structured Outputs**: Stream Pydantic-typed responses with `generate_pydantic_stream` or `agenerate_pydantic_stream`.
- **Async/Await Support**: All methods have asynchronous counterparts (e.g., `agenerate`, `adescribe_image`).
- **Structured Outputs**: Leverage Pydantic models for typed responses.
- **Image Description**: Generate descriptions for images with `describe_image`.
- **Code Execution**: Dynamically generate and execute Python code with `invoke_code_execution`.
- **Webpage to Markdown**: Convert webpages to Markdown with `PlayWrightBot`, including JavaScript rendering, element extraction, and LLM-based content filtering.
- **Citation Chunking**: Semantically chunk documents and extract citations with `citation_chunker`, including coverage analysis.
- **Interactive Shell**: Use `interactive_shell` for conversational AI with code execution.
- **Token Counting**: Retrieve input/output token counts with `get_num_tokens_from_message`.
- **Utilities**: Tools for content processing (e.g., `html_to_markdown`, `pdf_to_text`, `get_youtube_video_subtitle`, `citation_chunker`) are available in the `tools` module.

```python
# Example: Convert PDF to text
from chatterer.tools import pdf_to_text
text = pdf_to_text("example.pdf")
print(text)

# Example: Get YouTube subtitles
from chatterer.tools import get_youtube_video_subtitle
subtitles = get_youtube_video_subtitle("https://www.youtube.com/watch?v=example")
print(subtitles)

# Example: Get token counts
from chatterer.messages import HumanMessage
msg = HumanMessage(content="Hello, world!")
tokens = chatterer.get_num_tokens_from_message(msg)
if tokens:
    input_tokens, output_tokens = tokens
    print(f"Input: {input_tokens}, Output: {output_tokens}")
```

---

## Logging

Enable debugging with basic logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

The AoT pipeline uses a custom color-coded logger for detailed step-by-step output.

---

## Contributing

We welcome contributions! Feel free to open an issue or submit a pull request on the repository.

---

## License

MIT License
