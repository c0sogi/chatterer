import inspect
import textwrap
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    NamedTuple,
    Optional,
    Self,
)

from langchain_core.runnables.config import RunnableConfig

if TYPE_CHECKING:
    from langchain_experimental.tools import PythonAstREPLTool


class FunctionSignature(NamedTuple):
    name: str
    callable: Callable[..., object]
    signature: str

    @classmethod
    def from_callable(cls, callable: Callable[..., object]) -> Self:
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
    def from_callables(cls, callables: Iterable[Callable[..., object]]) -> list[Self]:
        return [cls.from_callable(callable) for callable in callables]

    @classmethod
    def as_prompt(
        cls,
        callables: Iterable[Self],
        prefix: Optional[str] = "You can use the pre-made functions below without defining them:\n",
        sep: str = "\n---\n",
    ) -> str:
        """
        Generate a prompt string from a list of callables.
        """
        body: str = sep.join(fsig.signature for fsig in callables)
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
        if function_signatures is not None:
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
        if function_signatures is not None:
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
