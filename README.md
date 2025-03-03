# Chatterer

**Simplified, Structured AI Assistant Framework**

`chatterer` is a Python library designed as a type-safe LangChain wrapper for interacting with various language models (OpenAI, Anthropic, Gemini, Ollama, etc.). It supports structured outputs via Pydantic models, plain text responses, and asynchronous calls.

The structured reasoning in `chatterer` is inspired by the [Atom-of-Thought](https://github.com/qixucen/atom) pipeline.

---

## Quick Install

```bash
pip install chatterer
```

---

## Quickstart Example

Generate text quickly using OpenAI:

```python
from chatterer import Chatterer

chat = Chatterer.openai("gpt-4o-mini")
response = chat.generate("What is the meaning of life?")
print(response)
```

Messages can be input as plain strings or structured lists:

```python
response = chat.generate([{ "role": "user", "content": "What's 2+2?" }])
print(response)
```

### Structured Output with Pydantic

```python
from pydantic import BaseModel

class AnswerModel(BaseModel):
    question: str
    answer: str

response = chat.generate_pydantic(AnswerModel, "What's the capital of France?")
print(response.question, response.answer)
```

### Async Example

```python
import asyncio

async def main():
    response = await chat.agenerate("Explain async in Python briefly.")
    print(response)

asyncio.run(main())
```

---

## Atom-of-Thought Pipeline (AoT)

`AoTPipeline` provides structured reasoning by:

- Detecting question domains (general, math, coding, philosophy, multihop).
- Decomposing questions recursively.
- Generating direct, decomposition-based, and simplified answers.
- Combining answers via ensemble.

### AoT Usage Example

```python
from chatterer import Chatterer
from chatterer.strategies import AoTStrategy, AoTPipeline

pipeline = AoTPipeline(chatterer=Chatterer.openai(), max_depth=2)
strategy = AoTStrategy(pipeline=pipeline)

question = "What would Newton discover if hit by an apple falling from 100 meters?"
answer = strategy.invoke(question)
print(answer)
```

---

## Supported Models

- **OpenAI**
- **Anthropic**
- **Google Gemini**
- **Ollama** (local models)

Initialize models easily:

```python
openai_chat = Chatterer.openai("gpt-4o-mini")
anthropic_chat = Chatterer.anthropic("claude-3-7-sonnet-20250219")
gemini_chat = Chatterer.google("gemini-2.0-flash")
ollama_chat = Chatterer.ollama("deepseek-r1:1.5b")
```

---

## Advanced Features

- **Streaming responses**
- **Async/Await support**
- **Structured outputs with Pydantic models**

---

## Logging

Built-in logging for easy debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Contributing

Feel free to open an issue or pull request.

---

## License

MIT License

