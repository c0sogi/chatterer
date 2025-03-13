#!/usr/bin/env python
import argparse
import json
import sys

sys.path.append(".")

from chatterer.tools.webpage_to_markdown.playwright_bot import PlayWrightBot


def save_session(url: str, jsonpath: str) -> None:
    """
    Launches a non-headless browser and navigates to the login_url.
    The user can manually log in, then press Enter in the console
    to store the current session state into a JSON file.
    """
    print(f"[*] Launching browser and navigating to {url} ... Please log in manually.")

    with PlayWrightBot(playwright_launch_options={"headless": False}) as bot:
        bot.get_page(url)

        print("[*] After completing the login in the browser, press Enter here to save the session.")
        input("    >> Press Enter when ready: ")

        # get_sync_browser() returns the BrowserContext internally
        context = bot.get_sync_browser()

        # Save the current session (cookies, localStorage) to a JSON file
        print(f"[*] Saving storage state to {jsonpath} ...")
        context.storage_state(path=jsonpath)

    print("[*] Done! Browser is now closed.")


def load_session(url: str, jsonpath: str) -> None:
    """
    Loads the session state from the specified JSON file, then navigates
    to a protected_url that normally requires login. If the stored session
    is valid, it should open without re-entering credentials.
    """
    print(f"[*] Loading session from {jsonpath} and navigating to {url} ...")

    with open(jsonpath, "r") as f:
        storage_state = json.load(f)

    with PlayWrightBot(
        playwright_launch_options={"headless": False}, playwright_persistency_options={"storage_state": storage_state}
    ) as bot:
        bot.get_page(url)

        print("[*] Press Enter in the console when you're done checking the protected page.")
        input("    >> Press Enter to exit: ")

    print("[*] Done! Browser is now closed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A simple CLI tool for saving and using Playwright sessions via storage_state."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: save
    parser_save = subparsers.add_parser("save", help="Save a new session by manually logging in.")
    parser_save.add_argument("url", help="URL to manually log in.")
    parser_save.add_argument(
        "--jsonpath",
        default=".tmp_session_storage.json",
        help="Path to save the session JSON file (default: .tmp_session_storage.json).",
    )

    # Subcommand: load
    parser_check = subparsers.add_parser("load", help="Use a saved session to view a protected page.")
    parser_check.add_argument("url", help="URL that requires login.")
    parser_check.add_argument(
        "--jsonpath",
        default=".tmp_session_storage.json",
        help="Path to the session JSON file to load (default: .tmp_session_storage.json).",
    )

    args = parser.parse_args()

    if args.command == "save":
        save_session(url=args.url, jsonpath=args.jsonpath)
    elif args.command == "load":
        load_session(url=args.url, jsonpath=args.jsonpath)


if __name__ == "__main__":
    main()
