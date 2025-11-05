# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportMissingTypeStubs=false

"""OpenRouter CLI tool for interacting with OpenRouter API."""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from loguru import logger
from python_open_router import OpenRouterClient
from python_open_router.models import CreatedKey
from spargear import RunnableArguments, SubcommandArguments, SubcommandSpec

BLUE = "\033[94m"
END = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"


class ListModelsArgs(RunnableArguments[None]):
    """Arguments for listing available models."""

    api_key: Optional[str] = None
    """API key for OpenRouter. Defaults to OPENROUTER_API_KEY environment variable."""
    details: bool = False
    """Whether to include detailed information about the models."""
    filter: Optional[list[str]] = None
    """Filter models by name, id, or description. Only models containing all of these strings will be listed."""

    def run(self) -> None:
        """List all available models from OpenRouter."""
        asyncio.run(self.arun())

    async def arun(self) -> None:
        """List all available models from OpenRouter."""
        client = OpenRouterClient(api_key=get_api_key(self.api_key))
        try:
            async with client:
                models = await client.get_models()
                logger.info(f"Found {len(models)} models:")

                for model in models:
                    should_continue = False
                    if self.details:
                        attrs: list[str] = [model.name, model.id, model.description]
                    else:
                        attrs = [model.id, model.name]

                    if self.filter:
                        for filter_str in self.filter:
                            if not any(filter_str.lower() in attr.lower() for attr in attrs):
                                should_continue = True
                                break
                    if should_continue:
                        continue

                    if self.details:
                        print(f"  - {GREEN}{model.id}{END} ({BLUE}{model.name}{END})")
                        print(f"\t- [Description] {BLUE}{model.description}{END}")
                        print(f"\t- [Context Length] {BLUE}{model.context_length}{END}")
                        print(
                            f"\t- [Architecture.input_modalities] {BLUE}{', '.join(modality.value for modality in model.architecture.input_modalities)}{END}"
                        )
                        print(
                            f"\t- [Architecture.output_modalities] {BLUE}{', '.join(modality.value for modality in model.architecture.output_modalities)}{END}"
                        )
                        print(
                            f"\t- [Supported Parameters] {BLUE}{', '.join(param.value for param in model.supported_parameters)}{END}"
                        )
                        print(
                            f"\t- [Input Modalities] {BLUE}{', '.join(modality.value for modality in model.architecture.input_modalities)}{END}"
                        )
                        print(
                            f"\t- [Output Modalities] {BLUE}{', '.join(modality.value for modality in model.architecture.output_modalities)}{END}"
                        )
                        print("\n")
                    else:
                        print(f"  - {model.id}")
        finally:
            await client.close()


class ChatArgs(RunnableArguments[None]):
    """Arguments for chatting with OpenRouter models."""

    message: str
    """The message to send to the model."""
    model: str = "openai/gpt-5-mini"
    """The model to use for the chat."""
    api_key: Optional[str] = None
    """API key for OpenRouter. Defaults to OPENROUTER_API_KEY environment variable."""
    base_url: str = "https://openrouter.ai/api/v1"
    """The base URL for the OpenRouter API."""
    stream: bool = False
    """Whether to stream the response."""

    def run(self) -> None:
        """Send a chat message and print the response."""
        asyncio.run(self.arun())

    async def arun(self) -> None:
        """Send a chat message and print the response."""
        # python_open_router doesn't provide chat API, use OpenAI-compatible API
        from openai import AsyncOpenAI

        async with AsyncOpenAI(api_key=get_api_key(self.api_key), base_url=self.base_url) as client:
            if self.stream:
                stream = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": self.message}],
                    stream=True,
                )
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        print(chunk.choices[0].delta.content, end="", flush=True)
                print()
            else:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": self.message}],
                )
                if response.choices and response.choices[0].message.content:
                    print(response.choices[0].message.content)


def get_api_key(api_key: Optional[str] = None) -> str:
    """Get API key from parameter or environment variable."""
    if api_key:
        return api_key
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        return env_key
    raise ValueError("API key not provided. Please set OPENROUTER_API_KEY environment variable or pass --api-key.")


def get_provisioning_api_key(api_key: Optional[str] = None) -> str:
    """Get provisioning API key from parameter or environment variable."""
    if api_key:
        return api_key
    env_key = os.getenv("OPENROUTER_PROVISIONING_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "Provisioning API key not provided. Please set OPENROUTER_PROVISIONING_API_KEY environment variable or pass --api-key."
    )


class GetKeyDataArgs(RunnableArguments[None]):
    """Arguments for getting current key data."""

    api_key: Optional[str] = None
    """API key for OpenRouter. Defaults to OPENROUTER_API_KEY environment variable."""

    def run(self) -> None:
        """Get key data for the current API key."""
        asyncio.run(self.arun())

    async def arun(self) -> None:
        """Get key data for the current API key."""
        client = OpenRouterClient(api_key=get_api_key(self.api_key))
        try:
            async with client:
                key_data = await client.get_key_data()
                print(f"Label: {BLUE}{key_data.label}{END}")
                print(f"Usage: {BLUE}{key_data.usage}{END}")
                print(f"Is Provisioning Key: {BLUE}{key_data.is_provisioning_key}{END}")
                print(f"Limit Remaining: {BLUE}{key_data.limit_remaining}{END}")
                print(f"Is Free Tier: {BLUE}{key_data.is_free_tier}{END}")
        finally:
            await client.close()


class ListKeysArgs(RunnableArguments[None]):
    """Arguments for listing all keys."""

    api_key: Optional[str] = None
    """Provisioning API key for OpenRouter. Defaults to OPENROUTER_PROVISIONING_API_KEY environment variable."""

    def run(self) -> None:
        """List all keys."""
        asyncio.run(self.arun())

    async def arun(self) -> None:
        """List all keys."""
        client = OpenRouterClient(api_key=get_provisioning_api_key(self.api_key))
        try:
            async with client:
                keys = await client.get_keys()
                logger.info(f"Found {len(keys)} keys:")
                for key in keys:
                    print(f"  - {key.name} ({key.label})")
                    print(f"    Hash: {key.hash}")
                    print(f"    Disabled: {key.disabled}")
                    print(f"    Limit: {key.limit}")
                    print(f"    Usage: {key.usage}")
                    print()
        finally:
            await client.close()


class CreateKeyArgs(RunnableArguments[None]):
    """Arguments for creating a new key."""

    name: str
    """Name for the new key."""
    limit: Optional[float] = None
    """Optional limit for the new key."""
    api_key: Optional[str] = None
    """API key for OpenRouter. Defaults to OPENROUTER_API_KEY environment variable."""

    def run(self) -> None:
        """Create a new key."""
        asyncio.run(self.arun())

    async def arun(self) -> None:
        """Create a new key."""
        client = OpenRouterClient(api_key=get_api_key(self.api_key))
        try:
            async with client:
                created_key_raw = await client.create_key(name=self.name, limit=self.limit)
                # create_key returns CreatedKey which has a 'key' attribute
                created_key: CreatedKey = created_key_raw  # type: ignore[assignment]
                print("Key created successfully!")
                print(f"Name: {created_key.name}")
                print(f"Label: {created_key.label}")
                print(f"Key: {created_key.key}")
                print(f"Hash: {created_key.hash}")
                print(f"Limit: {created_key.limit}")
                print(f"Usage: {created_key.usage}")
                print(f"Disabled: {created_key.disabled}")
                print()
                print(f"⚠️  IMPORTANT: Save this key now - {created_key.key}")
        finally:
            await client.close()


class CreditsArgs(RunnableArguments[None]):
    """Arguments for getting credits information."""

    api_key: Optional[str] = None
    """API key for OpenRouter. Defaults to OPENROUTER_API_KEY environment variable."""

    def run(self) -> None:
        """Get remaining credits."""
        import requests

        url = "https://openrouter.ai/api/v1/credits"
        headers = {"Authorization": f"Bearer {get_api_key(self.api_key)}"}
        response = requests.get(url, headers=headers)
        data = response.json()["data"]
        total_credits = data["total_credits"]
        total_usage = data["total_usage"]
        remaining_credits = total_credits - total_usage
        print(f"Total credits: {BLUE}{total_credits}{END}")
        print(f"Total usage: {BLUE}{total_usage}{END}")
        print(f"Remaining credits: {BLUE}{remaining_credits}{END}")


class Arguments(SubcommandArguments):
    """OpenRouter CLI tool for interacting with OpenRouter API."""

    _models: SubcommandSpec[ListModelsArgs] = SubcommandSpec(
        name="models",
        argument_class=ListModelsArgs,
        help="List all available models from OpenRouter.",
    )
    _chat: SubcommandSpec[ChatArgs] = SubcommandSpec(
        name="chat",
        argument_class=ChatArgs,
        help="Send a chat message to a model.",
    )
    _keydata: SubcommandSpec[GetKeyDataArgs] = SubcommandSpec(
        name="keydata",
        argument_class=GetKeyDataArgs,
        help="Get key data for the current API key.",
    )
    _keys: SubcommandSpec[ListKeysArgs] = SubcommandSpec(
        name="keys",
        argument_class=ListKeysArgs,
        help="List all keys.",
    )
    _createkey: SubcommandSpec[CreateKeyArgs] = SubcommandSpec(
        name="createkey",
        argument_class=CreateKeyArgs,
        help="Create a new key.",
    )
    _credits: SubcommandSpec[CreditsArgs] = SubcommandSpec(
        name="credits",
        argument_class=CreditsArgs,
        help="Get credits information.",
    )
