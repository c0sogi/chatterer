from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, TypeVar

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .language_model import Chatterer
from .utils.code_agent import (
    DEFAULT_CODE_GENERATION_PROMPT,
    DEFAULT_FUNCTION_REFERENCE_PREFIX_PROMPT,
    DEFAULT_FUNCTION_REFERENCE_SEPARATOR,
    CodeExecutionResult,
    FunctionSignature,
    augment_prompt_for_toolcall,
    get_default_repl_tool,
)

if TYPE_CHECKING:
    # Import only for type hinting to avoid circular dependencies if necessary
    from langchain_experimental.tools.python.tool import PythonAstREPLTool

T = TypeVar("T", bound=BaseModel)

# --- Pydantic Models ---


class ThinkBeforeSpeak(BaseModel):
    """
    Analyze the user's request and formulate an initial plan.
    This involves understanding the core task and breaking it down into logical steps.
    """

    task: str = Field(description="A concise summary of the user's overall goal or question.")
    plans: list[str] = Field(
        description="A sequence of actionable steps required to address the user's task. "
        "Each step should be clear and logical. Indicate if a step likely requires code execution."
    )


class IsToolCallNeeded(BaseModel):
    """
    Determine if executing Python code is the necessary *next* action.
    Carefully review the most recent messages, especially the last code execution output and review (if any).
    """

    is_tool_call_needed: bool = Field(
        description="Set to True ONLY if the *next logical step* requires executing Python code AND the previous step (if it involved code) did not already attempt this exact action and fail or produce unusable results. If the last code execution failed to achieve its goal (e.g., wrong data, error), set to False unless you plan to execute *different* code to overcome the previous issue. Set to False if the next step is reasoning, asking questions, or formulating a response based on existing information (including failed tool attempts)."
    )


class ReviewOnToolcall(BaseModel):
    """
    Evaluate the outcome of the Python code execution and decide the subsequent action.
    Critically assess if the execution achieved the intended goal and if the output is usable.
    """

    review_on_code_execution: str = Field(
        description="A critical analysis of the code execution result. Did it succeed technically? Did it produce the *expected and usable* output according to the plan? Explicitly mention any errors, unexpected values (like incorrect dates), or unusable results."
    )
    next_action: str = Field(
        description="Describe the *immediate next logical action* based on the review. **If the execution failed or yielded unusable/unexpected results, DO NOT suggest repeating the exact same code execution.** Instead, propose a different action, such as: 'Try a different code approach to get the time', 'Inform the user about the environmental issue with the date', 'Ask the user to verify the result', or 'Abandon this approach and try something else'. If the execution was successful and useful, describe the next step in the plan (e.g., 'Use the retrieved time to formulate the answer')."
    )
    is_task_completed: bool = Field(
        description="Set to True ONLY IF the *overall user task* is now fully addressed OR if the *only remaining action* based on the review is to generate the final response/answer directly to the user (this includes informing the user about an unresolvable issue found during execution). Set to False if further *productive* intermediate steps (like trying different code, processing data further, asking for input) are needed before the final response."
    )


class Think(BaseModel):
    """
    Engage in reasoning when code execution is not the immediate next step.
    This could involve synthesizing information, preparing the final answer, or identifying missing information.
    """

    my_thinking: str = Field(
        description="Explain your reasoning process. Why is code execution not needed now? "
        "What information are you using from the context? How are you planning to formulate the response or proceed?"
    )
    next_action: str = Field(
        description="Describe the *immediate next action* resulting from this thinking process. "
        "Examples: 'Formulate the final answer to the user', 'Ask the user a clarifying question', "
        "'Summarize the findings so far'."
    )
    # --- MODIFIED DESCRIPTION ---
    is_task_completed: bool = Field(
        description="Set this to True IF AND ONLY IF the 'next_action' you just described involves generating the final response, explanation, or answer directly for the user, based on the reasoning in 'my_thinking'. If the 'next_action' involves asking the user a question, planning *further* internal steps (beyond formulating the immediate response), or indicates the task cannot be completed yet, set this to False. **If the plan is simply to tell the user the answer now, set this to True.**"
    )
    # --- END OF MODIFICATION ---


# --- Interactive Shell Function ---


def interactive_shell(
    chatterer: Chatterer,
    system_instruction: BaseMessage | Iterable[BaseMessage] = ([
        SystemMessage(
            "You are an AI assistant capable of answering questions and executing Python code to help users solve tasks."
        ),
    ]),
    repl_tool: Optional["PythonAstREPLTool"] = None,
    prompt_for_code_invoke: Optional[str] = DEFAULT_CODE_GENERATION_PROMPT,
    additional_callables: Optional[Callable[..., object] | Iterable[Callable[..., object]]] = None,
    function_reference_prefix: Optional[str] = DEFAULT_FUNCTION_REFERENCE_PREFIX_PROMPT,
    function_reference_seperator: str = DEFAULT_FUNCTION_REFERENCE_SEPARATOR,
    config: Optional[RunnableConfig] = None,
    stop: Optional[list[str]] = None,
    **kwargs: Any,
) -> None:
    try:
        console = Console()
        # Style settings
        AI_STYLE = "bold bright_blue"
        EXECUTED_CODE_STYLE = "bold bright_yellow"
        OUTPUT_STYLE = "bold bright_cyan"
        THINKING_STYLE = "dim white"
    except ImportError:
        raise ImportError("Rich library not found. Please install it: pip install rich")

    # --- Shell Initialization and Main Loop ---
    if repl_tool is None:
        repl_tool = get_default_repl_tool()

    def set_locals(**kwargs: object) -> None:
        """Set local variables for the REPL tool."""
        if repl_tool.locals is None:  # pyright: ignore[reportUnknownMemberType]
            repl_tool.locals = {}
        for key, value in kwargs.items():
            repl_tool.locals[key] = value  # pyright: ignore[reportUnknownMemberType]

    def respond(messages: list[BaseMessage]) -> str:
        response = ""
        with console.status("[bold yellow]AI is thinking..."):
            response_panel = Panel("", title="AI Response", style=AI_STYLE, border_style="blue")
            current_content = ""
            for chunk in chatterer.generate_stream(messages=messages):
                current_content += chunk
                # Update renderable (might not display smoothly without Live)
                response_panel.renderable = current_content
            response = current_content
        console.print(Panel(response, title="AI Response", style=AI_STYLE))
        return response.strip()

    def complete_task(think_before_speak: ThinkBeforeSpeak) -> None:
        task_info = f"[bold]Task:[/bold] {think_before_speak.task}\n[bold]Plans:[/bold]\n- " + "\n- ".join(
            think_before_speak.plans
        )
        console.print(Panel(task_info, title="Task Analysis & Plan", style="magenta"))
        session_messages: list[BaseMessage] = [
            AIMessage(
                content=f"Okay, I understand the task. Here's my plan:\n"
                f"- Task Summary: {think_before_speak.task}\n"
                f"- Steps:\n" + "\n".join(f"  - {p}" for p in think_before_speak.plans)
            )
        ]

        while True:
            current_context = context + session_messages
            is_tool_call_needed: IsToolCallNeeded = chatterer(
                augment_prompt_for_toolcall(
                    function_signatures=function_signatures,
                    messages=current_context,
                    prompt_for_code_invoke=prompt_for_code_invoke,
                    function_reference_prefix=function_reference_prefix,
                    function_reference_seperator=function_reference_seperator,
                ),
                IsToolCallNeeded,
                config=config,
                stop=stop,
                **kwargs,
            )

            if is_tool_call_needed.is_tool_call_needed:
                # --- Code Execution Path ---
                set_locals(__context__=context, __session__=session_messages)
                code_execution: CodeExecutionResult = chatterer.exec(
                    messages=current_context,
                    repl_tool=repl_tool,
                    prompt_for_code_invoke=prompt_for_code_invoke,
                    function_signatures=function_signatures,
                    function_reference_prefix=function_reference_prefix,
                    function_reference_seperator=function_reference_seperator,
                    config=config,
                    stop=stop,
                    **kwargs,
                )
                code_block_display = (
                    f"[bold]Executed Code:[/bold]\n```python\n{code_execution.code}\n```\n\n"
                    f"[bold]Output:[/bold]\n{code_execution.output}"
                )
                console.print(
                    Panel(code_block_display, title="Code Execution", style=EXECUTED_CODE_STYLE, border_style="yellow")
                )
                tool_call_message = AIMessage(
                    content=f"I executed the following code:\n```python\n{code_execution.code}\n```\n**Output:**\n{code_execution.output}"
                )
                session_messages.append(tool_call_message)

                # --- Review Code Execution ---
                current_context_after_exec = context + session_messages
                decision = chatterer(
                    augment_prompt_for_toolcall(
                        function_signatures=function_signatures,
                        messages=current_context_after_exec,
                        prompt_for_code_invoke=prompt_for_code_invoke,
                        function_reference_prefix=function_reference_prefix,
                        function_reference_seperator=function_reference_seperator,
                    ),
                    ReviewOnToolcall,
                    config=config,
                    stop=stop,
                    **kwargs,
                )
                review_text = (
                    f"[bold]Review:[/bold] {decision.review_on_code_execution.strip()}\n"
                    f"[bold]Next Action:[/bold] {decision.next_action.strip()}"
                )
                console.print(Panel(review_text, title="Execution Review", style=OUTPUT_STYLE, border_style="cyan"))
                review_message = AIMessage(
                    content=f"**Review of Execution:** {decision.review_on_code_execution.strip()}\n"
                    f"**Next Action:** {decision.next_action.strip()}"
                )
                session_messages.append(review_message)

                # --- Check Completion after Review ---
                if decision.is_task_completed:
                    console.print(
                        Panel("[bold green]Task Completed![/bold green]", title="Status", border_style="green")
                    )
                    break  # Exit loop
            else:
                # --- Thinking Path (No Code Needed) ---
                current_context_before_think = context + session_messages
                decision = chatterer(
                    augment_prompt_for_toolcall(
                        function_signatures=function_signatures,
                        messages=current_context_before_think,
                        prompt_for_code_invoke=prompt_for_code_invoke,
                        function_reference_prefix=function_reference_prefix,
                        function_reference_seperator=function_reference_seperator,
                    ),
                    Think,
                    config=config,
                    stop=stop,
                    **kwargs,
                )
                thinking_text = (
                    f"[dim]Reasoning:[/dim] {decision.my_thinking.strip()}\n"
                    f"[bold]Next Action:[/bold] {decision.next_action.strip()}"
                )
                console.print(
                    Panel(
                        thinking_text, title="AI Thought Process (No Code)", style=THINKING_STYLE, border_style="white"
                    )
                )
                thinking_message = AIMessage(
                    content=f"**My Reasoning (without code execution):** {decision.my_thinking.strip()}\n"
                    f"**Next Action:** {decision.next_action.strip()}"
                )
                session_messages.append(thinking_message)

                # --- Check Completion after Thinking ---
                # This check now relies on the LLM correctly interpreting the updated
                # description for Think.is_task_completed
                if decision.is_task_completed:
                    console.print(
                        Panel("[bold green]Task Completed![/bold green]", title="Status", border_style="green")
                    )
                    break  # Exit loop

        # --- End of Loop ---
        # Generate and display the final response based on the *entire* session history
        final_response_messages = context + session_messages
        response: str = respond(final_response_messages)
        # Add the final AI response to the main context
        context.append(AIMessage(content=response))

    if additional_callables:
        if callable(additional_callables):
            additional_callables = [additional_callables]

        function_signatures: list[FunctionSignature] = FunctionSignature.from_callable(list(additional_callables))
    else:
        function_signatures: list[FunctionSignature] = []

    context: list[BaseMessage] = []
    if system_instruction:
        if isinstance(system_instruction, BaseMessage):
            context.append(system_instruction)
        elif isinstance(system_instruction, str):
            context.append(SystemMessage(content=system_instruction))
        else:
            context.extend(list(system_instruction))

    console.print(
        Panel(
            "Welcome to the Interactive Chatterer Shell!\nType 'quit' or 'exit' to end the conversation.",
            title="Welcome",
            style=AI_STYLE,
            border_style="blue",
        )
    )

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")
        except EOFError:
            user_input = "exit"

        if user_input.strip().lower() in ["quit", "exit"]:
            console.print(Panel("Goodbye!", title="Exit", style=AI_STYLE, border_style="blue"))
            break

        context.append(HumanMessage(content=user_input.strip()))

        try:
            # Initial planning step
            initial_plan_decision = chatterer(
                augment_prompt_for_toolcall(
                    function_signatures=function_signatures,
                    messages=context,
                    prompt_for_code_invoke=prompt_for_code_invoke,
                    function_reference_prefix=function_reference_prefix,
                    function_reference_seperator=function_reference_seperator,
                ),
                ThinkBeforeSpeak,
                config=config,
                stop=stop,
                **kwargs,
            )
            # Execute the task completion loop
            complete_task(initial_plan_decision)

        except Exception as e:
            import traceback

            console.print(
                Panel(
                    f"[bold red]An error occurred:[/bold red]\n{e}\n\n[yellow]Traceback:[/yellow]\n{traceback.format_exc()}",
                    title="Error",
                    border_style="red",
                )
            )


if __name__ == "__main__":
    interactive_shell(chatterer=Chatterer.openai())
