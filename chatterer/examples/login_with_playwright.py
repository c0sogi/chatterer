def resolve_import_path_and_get_logger():
    # ruff: noqa: E402
    import logging
    import sys

    if __name__ == "__main__" and "." not in sys.path:
        sys.path.append(".")

    logger = logging.getLogger(__name__)
    return logger


logger = resolve_import_path_and_get_logger()
import json
import sys
from pathlib import Path

from spargear import BaseArguments, SubcommandSpec

from chatterer import PlayWrightBot


def save_session(url: str, jsonpath: Path) -> None:
    """
    Launches a non-headless browser and navigates to the login_url.
    The user can manually log in, then press Enter in the console
    to store the current session state into a JSON file.
    """
    print(f"[*] Launching browser and navigating to {url} ... Please log in manually.")

    # Ensure jsonpath directory exists
    jsonpath.parent.mkdir(parents=True, exist_ok=True)

    with PlayWrightBot(playwright_launch_options={"headless": False}) as bot:
        bot.get_page(url)

        print("[*] After completing the login in the browser, press Enter here to save the session.")
        input("    >> Press Enter when ready: ")

        # get_sync_browser() returns the BrowserContext internally
        context = bot.get_sync_browser()

        # Save the current session (cookies, localStorage) to a JSON file
        print(f"[*] Saving storage state to {jsonpath} ...")
        context.storage_state(path=jsonpath)  # Pass Path object directly

    print("[*] Done! Browser is now closed.")


def load_session(url: str, jsonpath: Path) -> None:
    """
    Loads the session state from the specified JSON file, then navigates
    to a protected_url that normally requires login. If the stored session
    is valid, it should open without re-entering credentials.

    Correction: Loads the JSON content into a dict first to satisfy type hints.
    """
    print(f"[*] Loading session from {jsonpath} and navigating to {url} ...")

    if not jsonpath.exists():
        print(f"[!] Error: Session file not found at {jsonpath}")
        sys.exit(1)

    # Load the storage state from the JSON file into a dictionary
    print(f"[*] Reading storage state content from {jsonpath} ...")
    try:
        with open(jsonpath, "r", encoding="utf-8") as f:
            # This dictionary should match the 'StorageState' type expected by Playwright/chatterer
            storage_state_dict = json.load(f)
    except json.JSONDecodeError:
        print(f"[!] Error: Failed to decode JSON from {jsonpath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[!] Error reading file {jsonpath}: {e}", file=sys.stderr)
        sys.exit(1)

    print("[*] Launching browser with loaded session state...")
    with PlayWrightBot(
        playwright_launch_options={"headless": False},
        # Pass the loaded dictionary, which should match the expected 'StorageState' type
        playwright_persistency_options={"storage_state": storage_state_dict},
    ) as bot:
        bot.get_page(url)

        print("[*] Press Enter in the console when you're done checking the protected page.")
        input("    >> Press Enter to exit: ")

    print("[*] Done! Browser is now closed.")


# --- Spargear Declarative CLI Definition ---

# Define the default path location relative to this script file
DEFAULT_JSON_PATH = Path(__file__).resolve().parent / "session_state.json"


class SaveArgs(BaseArguments):
    """Arguments for the 'save' subcommand."""

    url: str
    """URL to navigate to for manual login."""
    jsonpath: Path = DEFAULT_JSON_PATH
    """Path to save the session state JSON file."""


class LoadArgs(BaseArguments):
    """Arguments for the 'load' subcommand."""

    url: str
    """URL (potentially protected) to navigate to using the saved session."""
    jsonpath: Path = DEFAULT_JSON_PATH
    """Path to the session state JSON file to load."""


class CliArgs(BaseArguments):
    """
    A simple CLI tool for saving and using Playwright sessions via storage_state.
    Uses spargear for declarative argument parsing.
    """

    save: SubcommandSpec[SaveArgs] = SubcommandSpec(
        name="save",
        argument_class=SaveArgs,
        help="Save a new session by manually logging in.",
        description="Launches a browser to the specified URL. Log in manually, then press Enter to save session state.",
    )
    load: SubcommandSpec[LoadArgs] = SubcommandSpec(
        name="load",
        argument_class=LoadArgs,
        help="Use a saved session to view a protected page.",
        description="Loads session state from the specified JSON file and navigates to the URL.",
    )


# --- Main Execution Logic ---


def main() -> None:
    """Parses arguments using spargear and executes the corresponding command."""
    # BaseArguments.load() parses arguments based on the CliArgs definition
    # It returns an instance of the *executed subcommand's argument class* (SaveArgs or LoadArgs)
    # or potentially None/raises error if parsing fails or no subcommand is given (as it's required here).
    try:
        # Pass create_instance=True if you want CliArgs.load() to return the instance directly
        # Otherwise, access values via class attributes like SaveArgs.url, SaveArgs.jsonpath after load()
        # Using create_instance=True is generally more straightforward for subcommand logic.
        # --> Correction: `load` *does* return the instance by default based on the provided spargear code.
        parsed_args = CliArgs()

        if isinstance(parsed_args, SaveArgs):
            # Access attributes directly from the returned instance
            print(f"[*] Running SAVE command:")
            print(f"    URL: {parsed_args.url}")
            print(f"    JSON Path: {parsed_args.jsonpath}")
            save_session(url=parsed_args.url, jsonpath=parsed_args.jsonpath)
        elif isinstance(parsed_args, LoadArgs):
            # Access attributes directly from the returned instance
            print(f"[*] Running LOAD command:")
            print(f"    URL: {parsed_args.url}")
            print(f"    JSON Path: {parsed_args.jsonpath}")
            load_session(url=parsed_args.url, jsonpath=parsed_args.jsonpath)
        elif parsed_args is None:
            # This case might occur if subcommands were optional and none was provided,
            # or if there was a parsing issue not caught by SystemExit.
            print("[!] No command executed. Use -h or --help for usage information.", file=sys.stderr)
            # Optionally print help:
            # CliArgs.get_parser().print_help()
            sys.exit(1)
        else:
            # Should not happen with the current structure
            print(f"[!] Unexpected argument parsing result: {type(parsed_args)}", file=sys.stderr)
            sys.exit(1)

    except SystemExit as e:
        # Handle cases like -h/--help or argparse errors that exit
        sys.exit(e.code)
    except Exception as e:
        print(f"\n[!] An error occurred: {e}", file=sys.stderr)
        # from traceback import print_exc # Uncomment for full traceback
        # print_exc()                   # Uncomment for full traceback
        sys.exit(1)


if __name__ == "__main__":
    main()
