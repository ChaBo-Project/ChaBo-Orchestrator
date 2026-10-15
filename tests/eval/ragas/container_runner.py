"""
Starts/stops a disposable local orchestrator container for Mode B - pulls the exact pinned
image an instance uses, injects the rendered config via INSTANCE_CONFIG_DIR (the same env var
a real deployment sets) plus secrets, waits for it to come up, then tears it down - even on a
crash mid-run, via the async context manager below.

Bound to 127.0.0.1 only: never reachable beyond the machine running eval, regardless of what
backing services (Qdrant, embedding, LLM) it's configured to call out to.
"""
import asyncio
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Dict

import httpx

CONTAINER_IMAGE_BASE = "ghcr.io/chabo-project/chabo-rag-orchestrator"
CONTAINER_INTERNAL_PORT = 7860
CONTAINER_NAME = "chabo-eval-temp"


class ContainerError(RuntimeError):
    """Raised when docker pull/run/health-check fails."""


def _run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ContainerError(f"{' '.join(cmd)} failed:\n{result.stdout}\n{result.stderr}")
    return result


def start_container(chabo_tag: str, rendered_config_dir: Path, host_port: int, secrets: Dict[str, str]) -> None:
    """Pull and start the container, bound to 127.0.0.1:<host_port>. Raises ContainerError on failure."""
    image = f"{CONTAINER_IMAGE_BASE}:{chabo_tag}"
    _run(["docker", "pull", image])

    env_args = ["-e", "INSTANCE_CONFIG_DIR=/app/instance_config"]
    for key, value in secrets.items():
        env_args += ["-e", f"{key}={value}"]

    _run([
        "docker", "run", "-d", "--rm",
        "--name", CONTAINER_NAME,
        "-p", f"127.0.0.1:{host_port}:{CONTAINER_INTERNAL_PORT}",
        "-v", f"{rendered_config_dir}:/app/instance_config:ro",
        *env_args,
        image,
    ])


def stop_container() -> None:
    """Stop and remove the container - safe to call even if it's already gone/never started."""
    subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True, text=True)


async def wait_for_health(host_port: int, timeout_s: float = 90.0) -> None:
    """Poll /health until it responds, or raise ContainerError after timeout_s."""
    deadline = time.monotonic() + timeout_s
    last_error = "no attempt succeeded"
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(f"http://127.0.0.1:{host_port}/health", timeout=5.0)
                if response.status_code == 200:
                    return
                last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = str(exc)
            await asyncio.sleep(2.0)
    raise ContainerError(f"Container did not become healthy within {timeout_s}s: {last_error}")


@asynccontextmanager
async def temp_container(
    chabo_tag: str, rendered_config_dir: Path, host_port: int, secrets: Dict[str, str]
) -> AsyncIterator[str]:
    """
    Start a disposable orchestrator container, wait for it, yield its base URL, and always
    stop it on the way out - including on an exception, so a crash mid-run never leaves a
    container running.
    """
    start_container(chabo_tag, rendered_config_dir, host_port, secrets)
    try:
        await wait_for_health(host_port)
        yield f"http://127.0.0.1:{host_port}"
    finally:
        stop_container()
