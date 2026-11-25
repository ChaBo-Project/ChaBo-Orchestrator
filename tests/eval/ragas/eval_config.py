"""
Config helpers for the RAGAS eval CLI.

Deliberately independent of the orchestrator's src/ code: the eval runs in its own venv and
only talks to an instance over HTTP, so a change in src/components/utils.py must never be
able to break it.
"""
import configparser
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # .env in the current directory, same convention as the orchestrator


def load_config(path: str) -> configparser.ConfigParser:
    """Read an INI file. A missing file yields an empty config, so env vars alone still work."""
    config = configparser.ConfigParser(inline_comment_prefixes=("#",))
    config.read(path)
    return config


def get_config_value(
    config: configparser.ConfigParser, section: str, key: str, env_var: str, fallback: Any = None
) -> Any:
    """Env var > config file > fallback. Raises ValueError if the value is required and missing."""
    env_value = os.getenv(env_var)
    if env_value is not None:
        return env_value
    try:
        return config.get(section, key)
    except (configparser.NoSectionError, configparser.NoOptionError):
        if fallback is not None:
            return fallback
        raise ValueError(
            f"Missing [{section}] {key} in eval.cfg, and the env var {env_var} is not set."
        ) from None


def get_api_key(env_var: str) -> str:
    """The judge's API key from the environment. Raises RuntimeError if it is not set."""
    value = os.getenv(env_var)
    if not value:
        raise RuntimeError(f"Missing judge API key: set the env var {env_var}.")
    return value
