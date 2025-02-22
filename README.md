# chatterer

`chatterer` is a Python library that provides a unified interface for interacting with various Language Model (LLM) backends. It abstracts over different providers such as OpenAI, Anthropic, DeepSeek, Ollama, and Langchain, allowing you to generate completions, stream responses, and even validate outputs using Pydantic models.

---

## Features

- **Unified LLM Interface**  
  Define a common interface (`LLM`) for generating completions and streaming responses regardless of the underlying provider.

- **Multiple Backend Support**  
  Built-in support for:
  - **InstructorLLM**: Integrates with OpenAI, Anthropic, and DeepSeek.
  - **OllamaLLM**: Supports the Ollama model with optional streaming and formatting.
  - **LangchainLLM**: Leverages Langchain’s chat models with conversion utilities.

- **Pydantic Integration**  
  Easily validate and structure LLM responses by leveraging Pydantic models with methods like `generate_pydantic` and `generate_pydantic_stream`.

---

## Installation

Assuming `chatterer` is published on PyPI, install it via pip:

```bash
pip install chatterer
```

Alternatively, clone the repository and install manually:

```bash
git clone https://github.com/yourusername/chatterer.git
cd chatterer
pip install -r requirements.txt
```

---

## Usage

### Importing the Library

You can import the core components directly from `chatterer`:

```python
from chatterer import LLM, InstructorLLM, OllamaLLM, LangchainLLM
```

---

### Example 1: Using InstructorLLM with OpenAI

```python
from chatterer import InstructorLLM
from openai.types.chat import ChatCompletionMessageParam

# Create an instance for OpenAI using the InstructorLLM wrapper
llm = InstructorLLM.openai(call_kwargs={"model": "o3-mini"})

# Define a conversation message list
messages: list[ChatCompletionMessageParam] = [
    {"role": "user", "content": "Hello, how can I help you?"}
]

# Generate a completion
response = llm.generate(messages)
print("Response:", response)

# Stream the response incrementally
print("Streaming response:")
for chunk in llm.generate_stream(messages):
    print(chunk, end="")
```

---

### Example 2: Using OllamaLLM

```python
from chatterer import OllamaLLM
from openai.types.chat import ChatCompletionMessageParam

# Initialize an OllamaLLM instance with streaming enabled
llm = OllamaLLM(model="ollama-model", stream=True)

messages: list[ChatCompletionMessageParam] = [
    {"role": "user", "content": "Tell me a joke."}
]

# Generate and print the full response
print("Response:", llm.generate(messages))

# Stream the response chunk by chunk
print("Streaming response:")
for chunk in llm.generate_stream(messages):
    print(chunk, end="")
```

---

### Example 3: Using LangchainLLM

```python
from chatterer import LangchainLLM
from openai.types.chat import ChatCompletionMessageParam
# Ensure you have a Langchain chat model instance; for example:
from langchain_core.language_models.chat_models import BaseChatModel

client: BaseChatModel = ...  # Initialize your Langchain chat model here
llm = LangchainLLM(client=client)

messages: list[ChatCompletionMessageParam] = [
    {"role": "user", "content": "What is the weather like today?"}
]

# Generate a complete response
response = llm.generate(messages)
print("Response:", response)

# Stream the response
print("Streaming response:")
for chunk in llm.generate_stream(messages):
    print(chunk, end="")
```

---

### Example 4: Using Pydantic for Structured Outputs

```python
from pydantic import BaseModel
from chatterer import InstructorLLM
from openai.types.chat import ChatCompletionMessageParam

# Define a response model
class MyResponse(BaseModel):
    response: str

# Initialize the InstructorLLM instance
llm = InstructorLLM.openai()

messages: list[ChatCompletionMessageParam] = [
    {"role": "user", "content": "Summarize this text."}
]

# Generate a structured response using a Pydantic model
structured_response = llm.generate_pydantic(MyResponse, messages)
print("Structured Response:", structured_response.response)
```

---

## API Overview

### `LLM` (Abstract Base Class)

- **Methods:**
  - `generate(messages: Sequence[ChatCompletionMessageParam]) -> str`  
    Generate a complete text response from a list of messages.
  
  - `generate_stream(messages: Sequence[ChatCompletionMessageParam]) -> Iterator[str]`  
    Stream the response incrementally.
  
  - `generate_pydantic(response_model: Type[P], messages: Sequence[ChatCompletionMessageParam]) -> P`  
    Generate and validate the response using a Pydantic model.
  
  - `generate_pydantic_stream(response_model: Type[P], messages: Sequence[ChatCompletionMessageParam]) -> Iterator[P]`  
    (Optional) Stream validated responses as Pydantic models.

### `InstructorLLM`

- Factory methods to create instances with various backends:
  - `openai()`
  - `anthropic()`
  - `deepseek()`

### `OllamaLLM`

- Supports additional options such as:
  - `model`, `stream`, `format`, `tools`, `options`, `keep_alive`

### `LangchainLLM`

- Integrates with Langchain's BaseChatModel and converts messages to a compatible format.

---

## Contributing

Contributions are highly encouraged! If you find a bug or have a feature request, please open an issue or submit a pull request on the repository. When contributing, please ensure your code adheres to the existing style and passes all tests.

---

## License

This project is licensed under the MIT License.
