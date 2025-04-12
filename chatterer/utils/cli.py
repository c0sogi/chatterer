import argparse
import io
import warnings
from dataclasses import dataclass, field, fields
from typing import (
    IO,
    Callable,
    Generic,
    Iterable,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)
from typing import (
    Literal as TypingLiteral,  # Alias to avoid confusion
)

# --- Type Definitions ---
SUPPRESS_LITERAL_TYPE = TypingLiteral["==SUPPRESS=="]
SUPPRESS: SUPPRESS_LITERAL_TYPE = "==SUPPRESS=="
ACTION_TYPES_THAT_DONT_SUPPORT_TYPE_KWARG = (
    "store_const",
    "store_true",
    "store_false",
    "append_const",
    "count",
    "help",
    "version",
)
Action = Optional[
    TypingLiteral[
        "store",
        "store_const",
        "store_true",
        "store_false",
        "append",
        "append_const",
        "count",
        "help",
        "version",
        "extend",
    ]
]
T = TypeVar("T")


@dataclass
class ArgumentSpec(Generic[T]):
    """Represents the specification for a command-line argument."""

    name_or_flags: list[str]
    action: Action = None
    nargs: Optional[Union[int, TypingLiteral["*", "+", "?"]]] = None
    const: Optional[object] = None
    default: Optional[Union[T, SUPPRESS_LITERAL_TYPE]] = None
    choices: Optional[Sequence[T]] = None
    required: bool = False
    help: str = ""
    metavar: Optional[str] = None
    version: Optional[str] = None
    type: Optional[Union[Callable[[str], T], argparse.FileType]] = None
    value: Optional[T] = field(init=False, default=None)  # Parsed value stored here

    @property
    def value_not_none(self) -> T:
        """Returns the value, raising an error if it's None."""
        if self.value is None:
            raise ValueError(f"Value for {self.name_or_flags} is None.")
        return self.value

    def get_add_argument_kwargs(self) -> dict[str, object]:
        """Prepares keyword arguments for argparse.ArgumentParser.add_argument."""
        kwargs: dict[str, object] = {}
        argparse_fields: set[str] = {f.name for f in fields(self) if f.name not in ("name_or_flags", "value")}
        for field_name in argparse_fields:
            attr_value = getattr(self, field_name)
            if field_name == "default":
                if attr_value is None:
                    pass  # Keep default=None if explicitly set or inferred
                elif attr_value in get_args(SUPPRESS_LITERAL_TYPE):
                    kwargs[field_name] = argparse.SUPPRESS
                else:
                    kwargs[field_name] = attr_value
            elif attr_value is not None:
                if field_name == "type" and self.action in ACTION_TYPES_THAT_DONT_SUPPORT_TYPE_KWARG:
                    continue
                kwargs[field_name] = attr_value
        return kwargs


class BaseArguments:
    """Base class for defining arguments declaratively using ArgumentSpec."""

    _arg_specs: dict[str, ArgumentSpec[object]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        """
        Processes ArgumentSpec definitions in subclasses upon class creation.
        Automatically infers 'type' and 'choices' from type hints if possible.
        """
        super().__init_subclass__(**kwargs)
        cls._arg_specs = {}
        for current_cls in reversed(cls.__mro__):
            if current_cls is object or current_cls is BaseArguments:
                continue
            current_vars = vars(current_cls)
            try:
                hints = get_type_hints(current_cls, globalns=dict(current_vars), include_extras=True)
                for attr_name, attr_value in current_vars.items():
                    if isinstance(attr_value, ArgumentSpec):
                        attr_value = cast(ArgumentSpec[object], attr_value)
                        generic_outer_type = None  # The T in ArgumentSpec[T]
                        element_type = None  # The E in ArgumentSpec[list[E]] or Sequence[E]

                        if attr_name in hints:
                            hint_origin = get_origin(hints[attr_name])
                            hint_args = get_args(hints[attr_name])
                            if hint_origin is ArgumentSpec and hint_args:
                                generic_outer_type = hint_args[0]  # Extract T
                                outer_origin = get_origin(generic_outer_type)
                                if outer_origin in (list, Sequence) and get_args(generic_outer_type):
                                    element_type = get_args(generic_outer_type)[0]  # Extract E

                        # --- Automatic 'choices' assignment for Literal types ---
                        # Check only if choices are not explicitly set
                        if attr_value.choices is None:
                            literal_type_to_check = None
                            # Case 1: ArgumentSpec[Literal["A", "B"]]
                            if generic_outer_type:
                                outer_origin = get_origin(generic_outer_type)
                                if outer_origin is TypingLiteral:
                                    literal_type_to_check = generic_outer_type

                            # Case 2: ArgumentSpec[list[Literal["A", "B"]]] or Sequence[Literal["A", "B"]]
                            # Check if element_type was derived and its origin is Literal
                            if literal_type_to_check is None and element_type:
                                element_origin = get_origin(element_type)
                                if element_origin is TypingLiteral:
                                    literal_type_to_check = element_type  # Use the element type itself

                            # If we found a Literal type either directly or as an element type
                            if literal_type_to_check:
                                literal_args = get_args(literal_type_to_check)
                                if literal_args:  # Check if Literal has arguments
                                    attr_value.choices = literal_args  # Assign choices

                        # --- End automatic 'choices' logic ---

                        # --- Automatic 'type' assignment logic ---
                        if attr_value.type is None:
                            type_to_assign = None
                            # If it's list[E] or Sequence[E], use E as type
                            if element_type and isinstance(element_type, type):
                                type_to_assign = element_type
                            # If it's T (and not a list/sequence of literals handled above), use T as type
                            elif generic_outer_type and isinstance(generic_outer_type, type):
                                # Avoid overriding type if choices were set from Literal
                                if not (attr_value.choices and get_origin(generic_outer_type) is TypingLiteral):
                                    type_to_assign = generic_outer_type

                            # Special handling for FileType which isn't a standard 'type'
                            elif (
                                generic_outer_type
                                and isinstance(generic_outer_type, type)
                                and issubclass(generic_outer_type, IO)
                            ):
                                # Let explicit type=argparse.FileType() handle this, don't auto-assign IO
                                pass
                            elif element_type and isinstance(element_type, type) and issubclass(element_type, IO):
                                # Let explicit type=argparse.FileType() handle this
                                pass

                            # Assign the inferred type if found and not explicitly None
                            if type_to_assign is not None:
                                attr_value.type = type_to_assign
                        # --- End automatic 'type' logic ---

                        cls._arg_specs[attr_name] = attr_value
            except Exception as e:
                warnings.warn(f"Could not fully analyze type hints for {current_cls.__name__}: {e}", stacklevel=2)
                for attr_name, attr_value in current_vars.items():
                    if isinstance(attr_value, ArgumentSpec) and attr_name not in cls._arg_specs:
                        cls._arg_specs[attr_name] = attr_value

    # ... (rest of the BaseArguments class remains the same) ...
    @classmethod
    def iter_specs(cls) -> Iterable[tuple[str, ArgumentSpec[object]]]:
        """Iterates over the registered (attribute_name, ArgumentSpec) pairs."""
        yield from cls._arg_specs.items()

    @classmethod
    def get_parser(cls) -> argparse.ArgumentParser:
        """Creates and configures an ArgumentParser based on the defined ArgumentSpecs."""
        arg_parser = argparse.ArgumentParser(
            description=cls.__doc__,  # Use class docstring as description
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=False,  # Add custom help argument later
        )
        # Add standard help argument
        arg_parser.add_argument(
            "-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit."
        )
        # Add arguments to the parser based on registered ArgumentSpecs
        for key, spec in cls.iter_specs():
            kwargs = spec.get_add_argument_kwargs()
            # Determine if it's a positional or optional argument
            is_positional: bool = not any(name.startswith("-") for name in spec.name_or_flags)
            if is_positional:
                # For positional args: remove 'required' (implicit), let argparse derive 'dest'
                kwargs.pop("required", None)
                try:
                    arg_parser.add_argument(*spec.name_or_flags, **kwargs)  # pyright: ignore[reportArgumentType]
                except Exception as e:
                    # Provide informative error message
                    raise ValueError(
                        f"Error adding positional argument '{key}' with spec {spec.name_or_flags} and kwargs {kwargs}: {e}"
                    ) from e
            else:  # Optional argument
                try:
                    # For optional args: explicitly set 'dest' to the attribute name ('key')
                    arg_parser.add_argument(*spec.name_or_flags, dest=key, **kwargs)  # pyright: ignore[reportArgumentType]
                except Exception as e:
                    # Provide informative error message
                    raise ValueError(
                        f"Error adding optional argument '{key}' with spec {spec.name_or_flags} and kwargs {kwargs}: {e}"
                    ) from e
        return arg_parser

    @classmethod
    def load(cls, args: Optional[Sequence[str]] = None) -> None:
        """
        Parses command-line arguments and assigns the values to the corresponding ArgumentSpec instances.
        If 'args' is None, uses sys.argv[1:].
        """
        parser = cls.get_parser()
        try:
            parsed_args = parser.parse_args(args)
        except SystemExit as e:
            # Allow SystemExit (e.g., from --help) to propagate
            raise e
        # Assign parsed values from the namespace
        cls.load_from_namespace(parsed_args)

    @classmethod
    def load_from_namespace(cls, args: argparse.Namespace) -> None:
        """Assigns values from a parsed argparse.Namespace object to the ArgumentSpecs."""
        for key, spec in cls.iter_specs():
            # Determine the attribute name in the namespace
            # Positional args use their name, optionals use the 'dest' (which is 'key')
            is_positional = not any(name.startswith("-") for name in spec.name_or_flags)
            attr_name = spec.name_or_flags[0] if is_positional else key
            # Check if the attribute exists in the namespace
            if hasattr(args, attr_name):
                value = getattr(args, attr_name)
                # Assign the value unless it's the SUPPRESS sentinel
                if value is not argparse.SUPPRESS:
                    spec.value = value
            # else: If the attribute isn't in the namespace (e.g., optional arg not provided
            # and no default), spec.value retains its initial value (usually None).

    @classmethod
    def get_value(cls, key: str) -> Optional[object]:
        """Retrieves the parsed value for a specific argument by its attribute name."""
        if key in cls._arg_specs:
            return cls._arg_specs[key].value
        raise KeyError(f"Argument spec with key '{key}' not found.")

    @classmethod
    def get_all_values(cls) -> dict[str, Optional[object]]:
        """Returns a dictionary of all argument attribute names and their parsed values."""
        return {key: spec.value for key, spec in cls.iter_specs()}


# --- Main execution block (Example Usage) ---
if __name__ == "__main__":

    class MyArguments(BaseArguments):
        """Example argument parser demonstrating various features."""

        my_str_arg: ArgumentSpec[str] = ArgumentSpec(
            ["-s", "--string-arg"], default="Hello", help="A string argument.", metavar="TEXT"
        )
        my_int_arg: ArgumentSpec[int] = ArgumentSpec(
            ["-i", "--integer-arg"], required=True, help="A required integer argument."
        )
        verbose: ArgumentSpec[bool] = ArgumentSpec(
            ["-v", "--verbose"], action="store_true", help="Increase output verbosity."
        )
        # --- List<str> ---
        my_list_arg: ArgumentSpec[list[str]] = ArgumentSpec(
            ["--list-values"],
            nargs="+",
            help="One or more string values.",
            default=None,
        )
        # --- Positional IO ---
        input_file: ArgumentSpec[IO[str]] = ArgumentSpec(
            ["input_file"],
            type=argparse.FileType("r", encoding="utf-8"),
            help="Path to the input file (required).",
            metavar="INPUT_PATH",
        )
        output_file: ArgumentSpec[Optional[IO[str]]] = ArgumentSpec(
            ["output_file"],
            type=argparse.FileType("w", encoding="utf-8"),
            nargs="?",
            default=None,
            help="Optional output file path.",
            metavar="OUTPUT_PATH",
        )
        # --- Simple Literal (choices auto-detected) ---
        log_level: ArgumentSpec[TypingLiteral["DEBUG", "INFO", "WARNING", "ERROR"]] = ArgumentSpec(
            ["--log-level"],
            default="INFO",
            help="Set the logging level.",
        )
        # --- Literal + explicit choices (explicit wins) ---
        mode: ArgumentSpec[TypingLiteral["fast", "slow", "careful"]] = ArgumentSpec(
            ["--mode"],
            choices=["fast", "slow"],  # Explicit choices override Literal args
            default="fast",
            help="Operation mode.",
        )
        # --- List[Literal] (choices auto-detected) ---
        enabled_features: ArgumentSpec[list[TypingLiteral["CACHE", "LOGGING", "RETRY"]]] = ArgumentSpec(
            ["--features"],
            nargs="*",  # 0 or more features
            help="Enable specific features.",
            default=[],
        )
        # --- SUPPRESS default ---
        optional_flag: ArgumentSpec[str] = ArgumentSpec(
            ["--opt-flag"],
            default=SUPPRESS,
            help="An optional flag whose attribute might not be set.",
        )

    print("--- Initial State (Before Parsing) ---")
    parser_for_debug = MyArguments.get_parser()
    for k, s in MyArguments.iter_specs():
        print(f"{k}: value={s.value}, type={s.type}, choices={s.choices}")  # Check inferred choices

    dummy_input_filename = "temp_input_for_argparse_test.txt"
    try:
        with open(dummy_input_filename, "w", encoding="utf-8") as f:
            f.write("This is a test file.\n")
        print(f"\nCreated dummy input file: {dummy_input_filename}")
    except Exception as e:
        print(f"Warning: Could not create dummy input file '{dummy_input_filename}': {e}")

    # Example command-line arguments (Adjusted order)
    test_args = [
        dummy_input_filename,
        "-i",
        "42",
        "--log-level",
        "WARNING",
        "--mode",
        "slow",
        "--list-values",
        "apple",
        "banana",
        "--features",
        "CACHE",
        "RETRY",  # Test List[Literal]
    ]
    # test_args = ['--features', 'INVALID'] # Test invalid choice for List[Literal]
    # test_args = ['-h']

    try:
        print(f"\n--- Loading Arguments (Args: {test_args if test_args else 'from sys.argv'}) ---")
        MyArguments.load(test_args)
        print("\n--- Final Loaded Arguments ---")
        all_values = MyArguments.get_all_values()
        for key, value in all_values.items():
            value_type = type(value).__name__
            if isinstance(value, io.IOBase):
                try:
                    name = getattr(value, "name", "<unknown_name>")
                    mode = getattr(value, "mode", "?")
                    value_repr = f"<IO {name} mode='{mode}'>"
                except ValueError:
                    value_repr = "<IO object (closed)>"
            else:
                value_repr = repr(value)
            print(f"{key}: {value_repr} (Type: {value_type})")

        print("\n--- Accessing Specific Values ---")
        print(f"Features     : {MyArguments.get_value('enabled_features')}")  # Check List[Literal] value

        input_f = MyArguments.get_value("input_file")
        if isinstance(input_f, io.IOBase):
            try:
                print(f"\nReading from input file: {input_f.name}")  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
                input_f.close()
                print(f"Closed input file: {input_f.name}")  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            except Exception as e:
                print(f"Error processing input file: {e}")

    except SystemExit as e:
        print(f"\nExiting application (SystemExit code: {e.code}).")
    except FileNotFoundError as e:
        print(f"\nError: Required file not found: {e.filename}")
        parser_for_debug.print_usage()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        import os

        if os.path.exists(dummy_input_filename):
            try:
                os.remove(dummy_input_filename)
                print(f"\nRemoved dummy input file: {dummy_input_filename}")
            except Exception as e:
                print(f"Warning: Could not remove dummy file '{dummy_input_filename}': {e}")
