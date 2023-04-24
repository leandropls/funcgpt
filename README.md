# funcgpt: Python library for creating functions with OpenAI's GPT

funcgpt is an easy-to-use Python library that allows you to quickly create Python functions using the power of OpenAI's GPT models. With just a few lines of code, you can create functions that generate human-like responses, answer questions, or anything else that GPT is capable of.

## Features

- Easy to use decorator for creating functions based on GPT models
- Supports different GPT model versions
- Customize GPT's behavior with adjustable temperature values
- Generate responses in streaming or non-streaming modes

## Installation

To install funcgpt, use pip:

```bash
pip install funcgpt
```

## Usage

To create a function that answers questions like a pirate, you can use the following snippet:

```python
from funcgpt import gpt

@gpt
def answer_like_pirate(message: str) -> str:
    """Answer questions like a pirate."""
    ...

```

Usage:

```python
>>> answer_like_pirate("How are you doing today?")
"Arrr, I be doin' fine, matey."
```

To do the same thing, but with a function that streams responses, you can use the following snippet:

```python
from typing import Iterator
from funcgpt import gpt

@gpt
def stream_like_pirate(message: str) -> Iterator[str]:
    """Answers questions like a pirate."""
    ...

```

Usage:

```python
>>> for token in stream_like_pirate("How are you doing today?"):
...     print(token, end="", flush=True)
...
Arrr, I be doin' fine, matey.
```

For defining a function that returns a boolean value, you can use the following snippet:

```python
from funcgpt import gpt

@gpt
def is_pirate(message: str) -> bool:
    """Returns true if the message is from a pirate."""
    ...

```

Usage:

```python
>>> is_pirate("Arrr, I be doin' fine, matey.")
True
```

For choosing a different model or temperature, you can use the `model` and `temperature` keyword arguments:

```python
from funcgpt import gpt

@gpt(model="gpt-4", temperature=0)
def answer_like_pirate(message: str) -> str:
    """Answer questions like a pirate."""
    ...

```

## Contributing

We welcome contributions! Please feel free to fork the repository, make changes, and submit pull requests. If you have any questions or ideas, don't hesitate to open an issue.

## License

funcgpt is released under the MIT License. See the [LICENSE](LICENSE) file for more details.
