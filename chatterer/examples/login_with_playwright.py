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


def read_session(url: str, jsonpath: Path) -> None:
    """
    Loads the session state from the specified JSON file, then navigates
    to a protected_url that normally requires login. If the stored session
    is valid, it should open without re-entering credentials.

    Correction: Loads the JSON content into a dict first to satisfy type hints.
    """
    logger.info(f"Loading session from {jsonpath} and navigating to {url} ...")

    if not jsonpath.exists():
        logger.error(f"Session file not found at {jsonpath}")
        sys.exit(1)

    # Load the storage state from the JSON file into a dictionary
    logger.info(f"Reading storage state content from {jsonpath} ...")
    try:
        with open(jsonpath, "r", encoding="utf-8") as f:
            # This dictionary should match the 'StorageState' type expected by Playwright/chatterer
            storage_state_dict = json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from {jsonpath}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading file {jsonpath}: {e}")
        sys.exit(1)

    logger.info("Launching browser with loaded session state...")
    with PlayWrightBot(
        playwright_launch_options={"headless": False},
        # Pass the loaded dictionary, which should match the expected 'StorageState' type
        playwright_persistency_options={"storage_state": storage_state_dict},
    ) as bot:
        bot.get_page(url)

        logger.info("Press Enter in the console when you're done checking the protected page.")
        input("    >> Press Enter to exit: ")

    logger.info("Done! Browser is now closed.")


def write_session(url: str, jsonpath: Path) -> None:
    """
    Launches a non-headless browser and navigates to the login_url.
    The user can manually log in, then press Enter in the console
    to store the current session state into a JSON file.
    """
    logger.info(f"Launching browser and navigating to {url} ... Please log in manually.")

    # Ensure jsonpath directory exists
    jsonpath.parent.mkdir(parents=True, exist_ok=True)

    with PlayWrightBot(playwright_launch_options={"headless": False}) as bot:
        bot.get_page(url)

        logger.info("After completing the login in the browser, press Enter here to save the session.")
        input("    >> Press Enter when ready: ")

        # get_sync_browser() returns the BrowserContext internally
        context = bot.get_sync_browser()

        # Save the current session (cookies, localStorage) to a JSON file
        logger.info(f"Saving storage state to {jsonpath} ...")
        context.storage_state(path=jsonpath)  # Pass Path object directly

    logger.info("Done! Browser is now closed.")


# --- Spargear Declarative CLI Definition ---

# Define the default path location relative to this script file
DEFAULT_JSON_PATH = Path(__file__).resolve().parent / "session_state.json"


class ReadArgs(BaseArguments):
    """Arguments for the 'read' subcommand."""

    url: str
    """URL (potentially protected) to navigate to using the saved session."""
    jsonpath: Path = DEFAULT_JSON_PATH
    """Path to the session state JSON file to load."""


class WriteArgs(BaseArguments):
    """Arguments for the 'write' subcommand."""

    url: str
    """URL to navigate to for manual login."""
    jsonpath: Path = DEFAULT_JSON_PATH
    """Path to save the session state JSON file."""


class LoginWithPlaywrightArgs(BaseArguments):
    """
    A simple CLI tool for saving and using Playwright sessions via storage_state.
    Uses spargear for declarative argument parsing.
    """

    read: SubcommandSpec[ReadArgs] = SubcommandSpec(
        name="read",
        argument_class=ReadArgs,
        help="Use a saved session to view a protected page.",
        description="Loads session state from the specified JSON file and navigates to the URL.",
    )
    write: SubcommandSpec[WriteArgs] = SubcommandSpec(
        name="write",
        argument_class=WriteArgs,
        help="Save a new session by manually logging in.",
        description="Launches a browser to the specified URL. Log in manually, then press Enter to save session state.",
    )

    def run(self) -> None:
        """Parses arguments using spargear and executes the corresponding command."""
        try:
            if (read := self.read.argument_class).url:
                # Access attributes directly from the returned instance
                logger.info("Running READ command:")
                logger.info(f"    URL: {read.url}")
                logger.info(f"    JSON Path: {read.jsonpath}")
                read_session(url=read.url, jsonpath=read.jsonpath)
            elif (write := self.write.argument_class).url:
                # Access attributes directly from the returned instance
                logger.info("Running WRITE command:")
                logger.info(f"    URL: {write.url}")
                logger.info(f"    JSON Path: {write.jsonpath}")
                write_session(url=write.url, jsonpath=write.jsonpath)
            else:
                logger.error("No valid subcommand provided. Use 'read' or 'write'.")
                sys.exit(1)

        except SystemExit as e:
            # Handle cases like -h/--help or argparse errors that exit
            sys.exit(e.code)
        except Exception as e:
            logger.error(f"\nAn error occurred: {e}")
            # from traceback import print_exc # Uncomment for full traceback
            # print_exc()                   # Uncomment for full traceback
            sys.exit(1)


# --- Main Execution Logic ---


if __name__ == "__main__":
    LoginWithPlaywrightArgs().run()
