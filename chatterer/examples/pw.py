import json
import sys
from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from chatterer.tools.web2md import PlayWrightBot

app = typer.Typer(help="A simple CLI tool for saving and using Playwright sessions via storage_state.")


def generate_json_path() -> Path:
    return Path("session_state.json").resolve()


@app.command()
def read(
    url: str = typer.Argument(help="URL (potentially protected) to navigate to using the saved session."),
    json_path: Optional[Path] = typer.Option(None, "--json", "-j", help="Path to the session state JSON file to load."),
) -> None:
    """Use a saved session to view a protected page."""
    jsonpath = json_path or generate_json_path()
    logger.info(f"Loading session from {jsonpath} and navigating to {url} ...")

    if not jsonpath.exists():
        logger.error(f"Session file not found at {jsonpath}")
        sys.exit(1)

    # Load the storage state from the JSON file into a dictionary
    logger.info(f"Reading storage state content from {jsonpath} ...")
    try:
        with open(jsonpath, "r", encoding="utf-8") as f:
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
        playwright_persistency_options={"storage_state": storage_state_dict},
    ) as bot:
        bot.get_page(url)

        logger.info("Press Enter in the console when you're done checking the protected page.")
        input("    >> Press Enter to exit: ")

    logger.info("Done! Browser is now closed.")


@app.command()
def write(
    url: str = typer.Argument(help="URL to navigate to for manual login."),
    json_path: Optional[Path] = typer.Option(None, "--json", "-j", help="Path to save the session state JSON file."),
) -> None:
    """Save a new session by manually logging in."""
    jsonpath = json_path or generate_json_path()
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
