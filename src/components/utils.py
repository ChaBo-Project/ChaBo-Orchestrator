import os
import re
import configparser
import logging
logger = logging.getLogger(__name__)
import requests # Need this for _call_hf_endpoint, but we will define the function here
import httpx
import yaml
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present

INSTANCE_CONFIG_DIR_ENV = "INSTANCE_CONFIG_DIR"


def instance_config_dir() -> Optional[str]:
    """
    Resolve INSTANCE_CONFIG_DIR. Returns None if unset.

    Raises ValueError if set but not a real directory — that's a misconfiguration
    (typo'd path, unmounted volume), not "no instance config", and should fail loudly
    rather than silently running with generic defaults.
    """
    path = os.getenv(INSTANCE_CONFIG_DIR_ENV)
    if not path:
        return None
    if not os.path.isdir(path):
        raise ValueError(
            f"{INSTANCE_CONFIG_DIR_ENV}={path!r} is set but is not a directory. "
            "Unset it to run with generic defaults, or point it at a real instance config directory."
        )
    return path


def params_override_path() -> Optional[str]:
    """Path to params.override.cfg if INSTANCE_CONFIG_DIR is set and the file exists, else None."""
    base = instance_config_dir()
    if not base:
        return None
    path = os.path.join(base, "params.override.cfg")
    return path if os.path.isfile(path) else None


def load_instance_yaml() -> Dict:
    """
    Load instance.yaml (Tier-2 content: filters, db_context, blocklist,
    instance_guidelines) if present. Returns {} if INSTANCE_CONFIG_DIR is unset or
    instance.yaml is absent. Each consumer reads its own top-level key and supplies
    its own empty/generic fallback.

    Raises ValueError on malformed YAML (fail fast, same posture as
    rewriter.db_context.load_db_context).
    """
    base = instance_config_dir()
    if not base:
        return {}
    path = os.path.join(base, "instance.yaml")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse instance.yaml at {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(
            f"instance.yaml at {path} must be a mapping at the top level, got {type(data).__name__}"
        )
    return data


_PROMPT_OVERRIDE_SECTION_RE = re.compile(r"^##\s+(\S+)\s*$", re.MULTILINE)


def load_prompt_overrides() -> Dict[str, str]:
    """
    Parse prompt_overrides.md's `## section_name` headers into {section_name: text}.
    Returns {} if INSTANCE_CONFIG_DIR is unset or prompt_overrides.md is absent.

    Tier-1, "change at your own risk" — overrides the steps/rules portion of a
    prompt only; callers must keep the output schema/contract core-owned and never
    substitute it from here.
    """
    base = instance_config_dir()
    if not base:
        return {}
    path = os.path.join(base, "prompt_overrides.md")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    sections: Dict[str, str] = {}
    matches = list(_PROMPT_OVERRIDE_SECTION_RE.finditer(text))
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip()
    return sections


def getconfig(configfile_path: str) -> configparser.ConfigParser:
    """
    Reads the config file, layering an optional Tier-1 instance override
    (INSTANCE_CONFIG_DIR/params.override.cfg) on top if present — later file wins
    per [section][key]. Every caller of getconfig() gets this for free.
    """
    config = configparser.ConfigParser()
    paths = [configfile_path]
    override = params_override_path()
    if override:
        paths.append(override)
    read_ok = config.read(paths)
    if configfile_path not in read_ok:
        logger.error(f"Warning: Config file not found at {configfile_path}. Relying on environment variables.")
    return config


def get_auth_for_generator(provider: str) -> dict:
    """Get authentication configuration for different providers"""
    auth_configs = {
        "openai": {"api_key": os.getenv("OPENAI_API_KEY")},
        "huggingface": {"api_key": os.getenv("HF_TOKEN")},
        "anthropic": {"api_key": os.getenv("ANTHROPIC_API_KEY")},
        "cohere": {"api_key": os.getenv("COHERE_API_KEY")},
        "azure": {"api_key": os.getenv("AZURE_API_KEY")},
    }
    
    if provider not in auth_configs:
        logger.error(f"Unsupported provider: {provider}")
        raise ValueError(f"Unsupported provider: {provider}")
    
    auth_config = auth_configs[provider]
    api_key = auth_config.get("api_key")
    
    if not api_key:
        logger.error(f"Missing API key for provider '{provider}'. Please set the appropriate environment variable.")
        raise RuntimeError(f"Missing API key for provider '{provider}'. Please set the appropriate environment variable.")
    
    return auth_config



def get_config_value(config: configparser.ConfigParser, section: str, key: str, env_var: str, fallback: Any = None) -> Any:
    """
    Retrieves a config value, prioritizing: Environment Variable > Config File > Fallback.
    """
    # 1. Check Environment Variable (Highest Priority)
    env_value = os.getenv(env_var)
    if env_value is not None:
        return env_value
        
    # 2. Check Config File
    try:
        return config.get(section, key)
    except (configparser.NoSectionError, configparser.NoOptionError):
        # 3. Use Fallback
        if fallback is not None:
            return fallback
        
        # 4. Error if essential config is missing
        logger.error(f"Configuration missing: Required value for [{section}]{key} was not found ")
        logger.error(f"in 'params.cfg' and the environment variable {env_var} is not set.")
        raise ValueError(
            f"Configuration missing: Required value for [{section}]{key} was not found "
            f"in 'params.cfg' and the environment variable {env_var} is not set."
        )

def _call_hf_endpoint(url: str, token: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Helper for making authenticated requests to Hugging Face Endpoints."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    try:
        logger.info(f"Calling endpoint {url}")
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 503:
                logger.warning(f"HF Endpoint 503: Service overloaded/starting up at {url}")
                raise Exception("HF Service Unavailable (503)")
        elif response.status_code == 404:
            logger.error(f"HF Endpoint 404: Model not found at {url}")
            raise Exception("HF Model Not Found (404)")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling HF endpoint ({url}): {e}")
        raise

async def _acall_hf_endpoint(url: str, token: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Asynchronously calls a Hugging Face Inference Endpoint using httpx."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    # Use httpx.AsyncClient for asynchronous requests
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            logger.info(f"Async Calling endpoint {url}")
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 503:
                logger.warning(f"HF Endpoint 503: Service overloaded/starting up at {url}")
                raise Exception("HF Service Unavailable (503)")
            elif response.status_code == 404:
                logger.error(f"HF Endpoint 404: Model not found at {url}")
                raise Exception("HF Model Not Found (404)")
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"Error calling HF endpoint ({url}): {e}")
            raise


def build_conversation_context(messages: List, max_turns: int = 3, max_chars: int = 8000) -> str:
    """
    Build conversation context from structured messages to send to generator.
    Always keeps the first user and assistant messages, plus the last N turns.

    A "turn" is one user message + following assistant response.

    Args:
        messages: List of Message objects with 'role' and 'content' attributes
        max_turns: Maximum number of user-assistant exchange pairs to include (from the end)
        max_chars: Maximum total characters in context (default 8000)

    Returns:
        Formatted conversation context string like:
        "USER: query1\nASSISTANT: response1\n\nUSER: query2\nASSISTANT: response2"
    """
    if not messages:
        return ""

    context_parts = []
    char_count = 0
    msgs_included = 0

    # Always include the first user and assistant messages
    first_user_msg = None
    first_assistant_msg = None

    # Find first user and assistant messages
    for msg in messages:
        if msg.role == 'user' and first_user_msg is None:
            first_user_msg = msg
        elif msg.role == 'assistant' and first_assistant_msg is None:
            first_assistant_msg = msg
        if first_user_msg and first_assistant_msg:
            break

    # Add first messages if they exist
    if first_user_msg:
        msg_text = f"USER: {first_user_msg.content}"
        msg_chars = len(msg_text)
        if char_count + msg_chars <= max_chars:
            context_parts.append(msg_text)
            char_count += msg_chars
            msgs_included += 1

    if first_assistant_msg:
        msg_text = f"ASSISTANT: {first_assistant_msg.content}"
        msg_chars = len(msg_text)
        if char_count + msg_chars <= max_chars:
            context_parts.append(msg_text)
            char_count += msg_chars
            msgs_included += 1

    # Collect last N complete turns (user + assistant pairs)
    # Find the last N user messages and their corresponding assistant responses
    user_messages = [msg for msg in messages if msg.role == 'user']

    # Get the last N user messages (excluding the first one we already included)
    recent_user_messages = user_messages[1:][-max_turns:] if len(user_messages) > 1 else []

    turn_count = 0
    recent_messages = []

    # Process each recent user message and find its corresponding assistant response
    for user_msg in recent_user_messages:
        if turn_count >= max_turns:
            break

        # Find the assistant response that follows this user message
        user_index = messages.index(user_msg)
        assistant_msg = None

        # Look for the next assistant message after this user message
        for i in range(user_index + 1, len(messages)):
            if messages[i].role == 'assistant':
                assistant_msg = messages[i]
                break

        # Add user message
        user_text = f"USER: {user_msg.content}"
        user_chars = len(user_text)

        if char_count + user_chars > max_chars:
            logger.info(f"Stopping context build: would exceed max_chars ({max_chars})")
            break

        recent_messages.append(user_text)
        char_count += user_chars
        msgs_included += 1

        # Add assistant message if it exists
        if assistant_msg:
            assistant_text = f"ASSISTANT: {assistant_msg.content}"
            assistant_chars = len(assistant_text)

            if char_count + assistant_chars > max_chars:
                logger.info(f"Stopping context build: would exceed max_chars ({max_chars})")
                break

            recent_messages.append(assistant_text)
            char_count += assistant_chars
            msgs_included += 1

        turn_count += 1

    # Add recent messages to context
    context_parts.extend(recent_messages)

    context = "\n\n".join(context_parts)

    logger.debug(f"Built conversation context: {msgs_included} messages, {char_count} chars")
    return context