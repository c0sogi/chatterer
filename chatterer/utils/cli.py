import argparse
import io
import typing
import warnings
from dataclasses import dataclass, field, fields
from typing import (
    IO,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

# --- Type Definitions ---
SUPPRESS_LITERAL_TYPE = Literal["==SUPPRESS=="]
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
    Literal[
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

    name_or_flags: List[str]
    action: Action = None
    nargs: Optional[Union[int, Literal["*", "+", "?"]]] = None
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

    def get_add_argument_kwargs(self) -> Dict[str, object]:
        """Prepares keyword arguments for argparse.ArgumentParser.add_argument."""
        kwargs: Dict[str, object] = {}
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


class ArgumentSpecType(NamedTuple):
    T: object  # The T in ArgumentSpec[T]
    element_type: typing.Optional[typing.Type[object]]  # The E in ArgumentSpec[List[E]] or ArgumentSpec[Tuple[E]]

    @classmethod
    def from_hint(cls, hints: Dict[str, object], attr_name: str):
        if (
            (hint := hints.get(attr_name))
            and (hint_origin := get_origin(hint))
            and (hint_args := get_args(hint))
            and isinstance(hint_origin, type)
            and issubclass(hint_origin, ArgumentSpec)
        ):
            T: object = hint_args[0]  # Extract T
            element_type: typing.Optional[object] = None
            if isinstance(outer_origin := get_origin(T), type):
                if issubclass(outer_origin, list) and (args := get_args(T)):
                    element_type = args[0]  # Extract E
                elif issubclass(outer_origin, tuple) and (args := get_args(T)):
                    element_type = args  # Extract E
                else:
                    element_type = None  # The E in ArgumentSpec[List[E]] or Tuple[E, ...]
                if not isinstance(element_type, type):
                    element_type = None
            return cls(T=T, element_type=element_type)

    @property
    def choices(self) -> typing.Optional[Tuple[object, ...]]:
        # ArgumentSpec[Literal["A", "B"]] or ArgumentSpec[List[Literal["A", "B"]]]
        T_origin = get_origin(self.T)
        if (
            isinstance(T_origin, type)
            and (issubclass(T_origin, (list, tuple)))
            and (args := get_args(self.T))
            and (get_origin(arg := args[0]) is typing.Literal)
            and (literals := get_args(arg))
        ):
            return literals
        elif T_origin is typing.Literal and (args := get_args(self.T)):
            return args

    @property
    def type(self) -> typing.Optional[typing.Type[object]]:
        if self.element_type is not None:
            return self.element_type  # If it's List[E] or Sequence[E], use E as type
        elif self.T and isinstance(self.T, type):
            return self.T  # Use T as type

    @property
    def should_return_as_list(self) -> bool:
        """Determines if the argument should be returned as a list."""
        T_origin = get_origin(self.T)
        if isinstance(T_origin, type):
            if issubclass(T_origin, list):
                return True
        return False

    @property
    def should_return_as_tuple(self) -> bool:
        """Determines if the argument should be returned as a tuple."""
        T_origin = get_origin(self.T)
        if isinstance(T_origin, type):
            if issubclass(T_origin, tuple):
                return True
        return False

    @property
    def tuple_nargs(self) -> Optional[int]:
        if self.should_return_as_tuple and (args := get_args(self.T)) and Ellipsis not in args:
            return len(args)


class BaseArguments:
    """Base class for defining arguments declaratively using ArgumentSpec."""

    __argspec__: Dict[str, ArgumentSpec[object]]
    __argspectype__: Dict[str, ArgumentSpecType]

    def __init_subclass__(cls, **kwargs: object) -> None:
        """
        Processes ArgumentSpec definitions in subclasses upon class creation.
        Automatically infers 'type' and 'choices' from type hints if possible.
        """
        super().__init_subclass__(**kwargs)
        cls.__argspec__ = {}
        cls.__argspectype__ = {}
        for current_cls in reversed(cls.__mro__):
            if current_cls is object or current_cls is BaseArguments:
                continue
            current_vars = vars(current_cls)
            try:
                hints: Dict[str, object] = typing.get_type_hints(current_cls, globalns=dict(current_vars))
                for attr_name, attr_value in current_vars.items():
                    if isinstance(attr_value, ArgumentSpec):
                        attr_value = typing.cast(ArgumentSpec[object], attr_value)
                        if arguments_spec_type := ArgumentSpecType.from_hint(hints=hints, attr_name=attr_name):
                            cls.__argspectype__[attr_name] = arguments_spec_type
                            if attr_value.choices is None and (literals := arguments_spec_type.choices):
                                attr_value.choices = literals
                            if attr_value.type is None and (type := arguments_spec_type.type):
                                attr_value.type = type
                            if tuple_nargs := arguments_spec_type.tuple_nargs:
                                attr_value.nargs = tuple_nargs
                        cls.__argspec__[attr_name] = attr_value
            except Exception as e:
                warnings.warn(f"Could not fully analyze type hints for {current_cls.__name__}: {e}", stacklevel=2)
                for attr_name, attr_value in current_vars.items():
                    if isinstance(attr_value, ArgumentSpec) and attr_name not in cls.__argspec__:
                        cls.__argspec__[attr_name] = attr_value

    @classmethod
    def iter_specs(cls) -> Iterable[Tuple[str, ArgumentSpec[object]]]:
        """Iterates over the registered (attribute_name, ArgumentSpec) pairs."""
        yield from cls.__argspec__.items()

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
            if not hasattr(args, attr_name):
                continue

            value: object = getattr(args, attr_name)
            if value is argparse.SUPPRESS:
                continue

            # Assign the value unless it's the SUPPRESS sentinel
            if argument_spec_type := cls.__argspectype__.get(key):
                if argument_spec_type.should_return_as_list:
                    if isinstance(value, list):
                        value = typing.cast(List[object], value)
                    elif value is not None:
                        value = [value]
                elif argument_spec_type.should_return_as_tuple:
                    if isinstance(value, tuple):
                        value = typing.cast(Tuple[object, ...], value)
                    elif value is not None:
                        if isinstance(value, list):
                            value = tuple(typing.cast(List[object], value))
                        else:
                            value = (value,)
            spec.value = value

    @classmethod
    def get_value(cls, key: str) -> Optional[object]:
        """Retrieves the parsed value for a specific argument by its attribute name."""
        if key in cls.__argspec__:
            return cls.__argspec__[key].value
        raise KeyError(f"Argument spec with key '{key}' not found.")

    @classmethod
    def get_all_values(cls) -> Dict[str, Optional[object]]:
        """Returns a dictionary of all argument attribute names and their parsed values."""
        return {key: spec.value for key, spec in cls.iter_specs()}

    def __init__(self) -> None:
        self.load()


def get_args(t: object) -> Tuple[object, ...]:
    """Returns the arguments of a type or a generic type."""
    return typing.get_args(t)


def get_origin(t: object) -> typing.Optional[object]:
    """Returns the origin of a type or a generic type."""
    return typing.get_origin(t)


# --- Main execution block (Example Usage) ---
if __name__ == "__main__":

    class __Arguments(BaseArguments):
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
        my_list_arg: ArgumentSpec[List[str]] = ArgumentSpec(
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
        log_level: ArgumentSpec[Literal["DEBUG", "INFO", "WARNING", "ERROR"]] = ArgumentSpec(
            ["--log-level"],
            default="INFO",
            help="Set the logging level.",
        )
        # --- Literal + explicit choices (explicit wins) ---
        mode: ArgumentSpec[Literal["fast", "slow", "careful"]] = ArgumentSpec(
            ["--mode"],
            choices=["fast", "slow"],  # Explicit choices override Literal args
            default="fast",
            help="Operation mode.",
        )
        # --- List[Literal] (choices auto-detected) ---
        enabled_features: ArgumentSpec[List[Literal["CACHE", "LOGGING", "RETRY"]]] = ArgumentSpec(
            ["--features"],
            nargs="*",  # 0 or more features
            help="Enable specific features.",
            default=[],
        )
        tuple_features: ArgumentSpec[
            Tuple[Literal["CACHE", "LOGGING", "RETRY"], Literal["CAwCHE", "LOGGING", "RETRY"]]
        ] = ArgumentSpec(
            ["--tuple-features"],
            help="Enable specific features (tuple).",
        )

        # --- SUPPRESS default ---
        optional_flag: ArgumentSpec[str] = ArgumentSpec(
            ["--opt-flag"],
            default=SUPPRESS,
            help="An optional flag whose attribute might not be set.",
        )

    print("--- Initial State (Before Parsing) ---")
    parser_for_debug = __Arguments.get_parser()
    for k, s in __Arguments.iter_specs():
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
        "--tuple-features",
        "CACHE",
        "LOGGING",  # Test Tuple[Literal]
    ]
    # test_args = ['--features', 'INVALID'] # Test invalid choice for List[Literal]
    # test_args = ['-h']

    try:
        print(f"\n--- Loading Arguments (Args: {test_args if test_args else 'from sys.argv'}) ---")
        __Arguments.load(test_args)
        print("\n--- Final Loaded Arguments ---")
        all_values = __Arguments.get_all_values()
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
        print(f"Features     : {__Arguments.get_value('enabled_features')}")  # Check List[Literal] value

        input_f = __Arguments.get_value("input_file")
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
