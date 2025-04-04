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

# --- Constants ---
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


# --- Helper Classes and Functions ---


class FunctionSignature(NamedTuple):
    name: str
    callable: Callable[..., object]
    signature: str

    @classmethod
    def from_callable(cls, callables: Optional[Callable[..., object] | Iterable[Callable[..., object]]]) -> list[Self]:
        if callables is None:
            return []
        # Correctly handle single callable case
        if isinstance(callables, Callable) and not isinstance(callables, type):  # Exclude classes if not intended
            return [cls._from_callable(callables)]
        # Handle iterables
        if isinstance(callables, Iterable):
            return [cls._from_callable(c) for c in callables]
        # If it's neither a callable nor an iterable of callables, return empty
        return []

    @classmethod
    def _from_callable(cls, callable_obj: Callable[..., object]) -> Self:
        """
        Get the name and signature of a function as a string.
        """
        is_async_func = inspect.iscoroutinefunction(callable_obj)
        function_def = "async def" if is_async_func else "def"

        if inspect.isfunction(callable_obj):
            function_name = callable_obj.__code__.co_name
        elif hasattr(callable_obj, "name") and isinstance(getattr(callable_obj, "name"), str):
            function_name = getattr(callable_obj, "name")
        elif hasattr(callable_obj, "__name__"):
            function_name = callable_obj.__name__
        else:
            function_name = type(callable_obj).__name__

        try:
            signature_str = str(inspect.signature(callable_obj))
        except ValueError:  # Handles built-ins or others without inspectable signatures
            signature_str = "(...)"  # Placeholder signature

        signature = f"{function_def} {function_name}{signature_str}:"
        docstring = inspect.getdoc(callable_obj)

        if docstring:
            docstring = f'"""{docstring.strip()}"""'
            full_signature = f"{signature}\n{textwrap.indent(docstring, '    ')}"
        else:
            full_signature = signature

        return cls(name=function_name, callable=callable_obj, signature=full_signature)

    @classmethod
    def as_prompt(
        cls,
        function_signatures: Iterable[Self],
        prefix: Optional[str] = DEFAULT_FUNCTION_REFERENCE_PREFIX_PROMPT,  # Use constant
        sep: str = DEFAULT_FUNCTION_REFERENCE_SEPARATOR,  # Use constant
    ) -> str:
        """
        Generate a prompt string from a list of function signatures.
        """
        if not function_signatures:
            return ""
        body: str = sep.join(fsig.signature for fsig in function_signatures)
        if prefix:
            return f"{prefix}\n{body}"  # Add newline for clarity
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
        Execute code using the Python REPL tool.
        """
        if repl_tool is None:
            repl_tool = get_default_repl_tool()
        if function_signatures:
            insert_callables_into_global(function_signatures=function_signatures, repl_tool=repl_tool)
        # Ensure kwargs are passed correctly if needed by invoke
        output = str(repl_tool.invoke(code, config=config))  # pyright: ignore[reportUnknownMemberType]
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
        Execute code using the Python REPL tool asynchronously.
        """
        if repl_tool is None:
            repl_tool = get_default_repl_tool()
        if function_signatures:
            insert_callables_into_global(function_signatures=function_signatures, repl_tool=repl_tool)
        # Ensure kwargs are passed correctly if needed by ainvoke
        output = str(await repl_tool.ainvoke(code, config=config))  # pyright: ignore[reportUnknownMemberType]if not needed by base ainvoke
        return cls(code=code, output=output)


def get_default_repl_tool() -> "PythonAstREPLTool":
    """Initializes and returns a default PythonAstREPLTool instance."""
    try:
        from langchain_experimental.tools import PythonAstREPLTool

        # You might want to configure specific globals/locals here if needed
        return PythonAstREPLTool()
    except ImportError:
        raise ImportError(
            "PythonAstREPLTool requires langchain_experimental. Install with: pip install langchain-experimental"
        )


def insert_callables_into_global(
    function_signatures: Iterable[FunctionSignature], repl_tool: "PythonAstREPLTool"
) -> None:
    """Insert callables into the REPL tool's globals."""
    # Accessing globals might depend on the specific REPL tool implementation.
    # This assumes a .globals attribute exists and is a dict.
    if not hasattr(repl_tool, "globals") or not isinstance(repl_tool.globals, dict):  # pyright: ignore[reportUnknownMemberType]
        # Handle cases where .globals is not available or not a dict
        # Maybe initialize it or log a warning/error
        repl_tool.globals = {}  # Or handle appropriately

    # Safely update globals
    current_globals: dict[object, object] = repl_tool.globals  # pyright: ignore[reportUnknownMemberType]
    for fsig in function_signatures:
        current_globals[fsig.name] = fsig.callable
    # No need to reassign if globals is mutable (dict)
    # repl_tool.globals = current_globals


def _add_message_first(messages: LanguageModelInput, prompt_to_add: str) -> LanguageModelInput:
    """Prepends a SystemMessage to the beginning of the message list/string."""
    if not prompt_to_add:  # Don't add empty prompts
        return messages

    if isinstance(messages, str):
        # Prepend with a newline for separation
        return f"{prompt_to_add}\n\n{messages}"
    elif isinstance(messages, Sequence):
        # Create a mutable copy if it's a tuple
        msg_list = list(messages)
        msg_list.insert(0, SystemMessage(content=prompt_to_add))
        return msg_list
    # Handle LangChain Core BaseMessagePromptTemplate or similar if needed
    # elif hasattr(messages, 'to_messages'):
    #    msg_list = messages.to_messages()
    #    msg_list.insert(0, SystemMessage(content=prompt_to_add))
    #    return msg_list # Or return a new prompt template if required
    else:
        # Fallback or raise error for unsupported types
        raise TypeError(f"Unsupported message input type: {type(messages)}")


def augment_prompt_for_toolcall(
    function_signatures: Iterable[FunctionSignature],
    messages: LanguageModelInput,
    prompt_for_code_invoke: Optional[str] = DEFAULT_CODE_GENERATION_PROMPT,
    function_reference_prefix: Optional[str] = DEFAULT_FUNCTION_REFERENCE_PREFIX_PROMPT,
    function_reference_seperator: str = DEFAULT_FUNCTION_REFERENCE_SEPARATOR,
) -> LanguageModelInput:
    """Adds function references and code invocation prompts to the messages."""
    # Add function references first (if any)
    func_prompt = FunctionSignature.as_prompt(
        function_signatures, function_reference_prefix, function_reference_seperator
    )
    if func_prompt:
        messages = _add_message_first(messages=messages, prompt_to_add=func_prompt)

    # Then add the main code invocation prompt (if provided)
    if prompt_for_code_invoke:
        messages = _add_message_first(messages=messages, prompt_to_add=prompt_for_code_invoke)

    return messages
