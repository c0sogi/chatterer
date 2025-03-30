import inspect
import textwrap
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    NamedTuple,
    Optional,
    Self,
    Sequence,
)

from langchain_core.runnables.config import RunnableConfig

from ..messages import LanguageModelInput, SystemMessage

if TYPE_CHECKING:
    from langchain_experimental.tools import PythonAstREPLTool

DEFAULT_CODE_GENERATION_PROMPT = (
    "You are equipped with a Python code execution tool.\n"
    "Your primary goal is to generate Python code that effectively solves the *specific, immediate sub-task* required to progress towards the overall user request. The generated code and its resulting output will be automatically added to our conversation history.\n"
    "\n"
    "Guidelines for Optimal Tool Use:\n"
    "- Conciseness and Efficiency: Write code that directly addresses the current need. Avoid unnecessary complexity, computations, or data loading. Tool execution has resource limits.\n"
    "- Targeted Action: Focus only on the code required for the *next logical step*. Do not attempt to solve the entire problem in one code block if it involves multiple steps.\n"
    "- Error Handling: Implement basic error handling (e.g., `try-except`) for operations that might fail (like file access or network requests, if applicable).\n"
    "- Context Awareness: Assume the code runs in a stateful environment where variables and imports might persist from previous executions (unless explicitly cleared).\n"
    "- Self-Contained Execution: Ensure the code block is runnable as provided. Define necessary variables within the block if they aren't guaranteed to exist from prior context.\n"
    "\n"
    "Output Format:\n"
    "Return *only* a JSON object containing the Python code:\n"
    '{\n  "code": "<your_python_code_here>"\n}\n\n'
)

DEFAULT_FUNCTION_REFERENCE_PREFIX_PROMPT = (
    "The following Python functions are available in the global scope for you to use directly in your code.\n"
    "You do not need to define these functions; simply call them as needed.\n"
    "Use these functions only when they directly help in solving the current task. You are not obligated to use them.\n"
)
DEFAULT_FUNCTION_REFERENCE_SEPARATOR = "\n---\n"  # Separator to distinguish different function references


class FunctionSignature(NamedTuple):
    name: str
    callable: Callable[..., object]
    signature: str

    @classmethod
    def from_callable(cls, callables: Optional[Callable[..., object] | Iterable[Callable[..., object]]]) -> list[Self]:
        if callables is None:
            return []
        if callable(callables):
            return [cls._from_callable(callables)]
        return [cls._from_callable(callable) for callable in callables]

    @classmethod
    def _from_callable(cls, callable: Callable[..., object]) -> Self:
        """
        Get the name and signature of a function as a string.
        """
        # Determine if the function is async
        is_async_func = inspect.iscoroutinefunction(callable)
        function_def = "async def" if is_async_func else "def"

        # Determine the function name based on the type of callable
        if inspect.isfunction(callable):
            # For regular Python functions, use __code__.co_name
            function_name = callable.__code__.co_name
        elif hasattr(callable, "name"):
            # For StructuredTool or similar objects with a 'name' attribute
            function_name = callable.name  # type: ignore
        elif hasattr(callable, "__name__"):
            # For other callables with a __name__ attribute
            function_name = callable.__name__
        else:
            # Fallback to the class name if no name is found
            function_name = type(callable).__name__

        # Build the signature string
        signature = f"{function_def} {function_name}{inspect.signature(callable)}:"
        docstring = inspect.getdoc(callable)
        if docstring:
            docstring = f'"""{docstring.strip()}"""'
            return cls(
                name=function_name, callable=callable, signature=f"{signature}\n{textwrap.indent(docstring, '    ')}"
            )
        else:
            return cls(name=function_name, callable=callable, signature=signature)

    @classmethod
    def as_prompt(
        cls,
        function_signatures: Iterable[Self],
        prefix: Optional[str] = "You can use the pre-made functions below without defining them:\n",
        sep: str = "\n---\n",
    ) -> str:
        """
        Generate a prompt string from a list of callables.
        """
        body: str = sep.join(fsig.signature for fsig in function_signatures)
        if prefix:
            return f"{prefix}{body}"
        return body


class CodeExecutionResult(NamedTuple):
    code: str
    output: str

    @classmethod
    def from_code(
        cls,
        code: str,
        repl_tool: Optional["PythonAstREPLTool"] = None,
        config: Optional[RunnableConfig] = None,
        function_signatures: Optional[Iterable[FunctionSignature]] = None,
        **kwargs: object,
    ) -> Self:
        """
        Execute code using the Python Code Execution Language Model.
        """
        if repl_tool is None:
            repl_tool = get_default_repl_tool()
        if function_signatures:
            insert_callables_into_global(function_signatures=function_signatures, repl_tool=repl_tool)
        output = str(repl_tool.invoke(code, config=config, **kwargs))  # pyright: ignore[reportUnknownMemberType]
        return cls(code=code, output=output)

    @classmethod
    async def afrom_code(
        cls,
        code: str,
        repl_tool: Optional["PythonAstREPLTool"] = None,
        config: Optional[RunnableConfig] = None,
        function_signatures: Optional[Iterable[FunctionSignature]] = None,
        **kwargs: object,
    ) -> Self:
        """
        Execute code using the Python Code Execution Language Model asynchronously.
        """
        if repl_tool is None:
            repl_tool = get_default_repl_tool()
        if function_signatures:
            insert_callables_into_global(function_signatures=function_signatures, repl_tool=repl_tool)
        output = str(await repl_tool.ainvoke(code, config=config, **kwargs))  # pyright: ignore[reportUnknownMemberType]
        return cls(code=code, output=output)


def get_default_repl_tool() -> "PythonAstREPLTool":
    from langchain_experimental.tools import PythonAstREPLTool

    return PythonAstREPLTool()


def insert_callables_into_global(
    function_signatures: Iterable[FunctionSignature], repl_tool: "PythonAstREPLTool"
) -> None:
    """Insert callables into the REPL tool's globals."""
    repl_globals: Optional[dict[str, object]] = repl_tool.globals  # pyright: ignore[reportUnknownMemberType]
    if repl_globals is None:
        repl_tool.globals = {fsig.name: fsig.callable for fsig in function_signatures}
    else:
        repl_globals.update({fsig.name: fsig.callable for fsig in function_signatures})


def _add_message_first(messages: LanguageModelInput, prompt_to_add: str) -> LanguageModelInput:
    if isinstance(messages, str):
        messages = f"{prompt_to_add}\n{messages}"
    elif isinstance(messages, Sequence):
        messages = list(messages)
        messages.insert(0, SystemMessage(content=prompt_to_add))
    else:
        messages = messages.to_messages()
        messages.insert(0, SystemMessage(content=prompt_to_add))
    return messages


def augment_prompt_for_toolcall(
    function_signatures: Iterable[FunctionSignature],
    messages: LanguageModelInput,
    prompt_for_code_invoke: Optional[str] = DEFAULT_CODE_GENERATION_PROMPT,
    function_reference_prefix: Optional[str] = DEFAULT_FUNCTION_REFERENCE_PREFIX_PROMPT,
    function_reference_seperator: str = DEFAULT_FUNCTION_REFERENCE_SEPARATOR,
) -> LanguageModelInput:
    if function_signatures:
        messages = _add_message_first(
            messages=messages,
            prompt_to_add=FunctionSignature.as_prompt(
                function_signatures, function_reference_prefix, function_reference_seperator
            ),
        )
    if prompt_for_code_invoke:
        messages = _add_message_first(messages=messages, prompt_to_add=prompt_for_code_invoke)
    return messages
