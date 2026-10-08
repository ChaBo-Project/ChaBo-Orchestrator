"""Shared logging format for the Orchestrator."""

import logging
import os
import sys
import time
import asyncio
import uuid
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

run_id_context = ContextVar("chabo_log_run_id", default="-")

execution_log_context = ContextVar("chabo_execution_log", default=None)


@dataclass
class ExecutionLogState:
    handler: logging.Handler
    active: bool = True


class ExecutionFileRouter(logging.Handler):
    """Send each record only to its current execution's file."""

    def emit(self, record):
        state = execution_log_context.get()
        if state is None or not state.active:
            return
        record.run_id = run_id_context.get()
        state.handler.handle(record)

class ExecutionContextFilter(logging.Filter):
    def filter(self, record):
        record.run_id = run_id_context.get()
        return True


class ConsoleFormatter(logging.Formatter):
    converter = time.gmtime

    def __init__(self, use_color=False):
        super().__init__(
            fmt=(
                "[%(asctime)s.%(msecs)03dZ] [%(levelname)s] "
                "[PID=%(process)d] [RUN=%(run_id)s] "
                "[%(name)s] %(message)s"
            ),
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        self.use_color = use_color

    def format(self, record):
        text = super().format(record)
        if self.use_color and record.levelno >= logging.ERROR:
            return f"\033[31m{text}\033[0m"
        return text


def configure_logging():
    """Configure console output and a dedicated log file per process."""
    color_mode = os.getenv("LOG_COLOR", "auto").strip().lower()
    if color_mode not in {"auto", "always", "never"}:
        raise ValueError("LOG_COLOR must be auto, always, or never.")

    use_color = color_mode == "always" or (
        color_mode == "auto"
        and sys.stderr.isatty()
        and "NO_COLOR" not in os.environ
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.addFilter(ExecutionContextFilter())
    console_handler.setFormatter(ConsoleFormatter(use_color=use_color))

    process_id = os.getpid()
    process_instance = uuid.uuid4().hex
    base_dir = Path(os.getenv("LOG_DIR", "logs"))
    process_dir = base_dir / f"process-{process_id}-{process_instance}"
    process_dir.mkdir(parents=True, exist_ok=False)

    process_handler = RotatingFileHandler(
        process_dir / "process.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    process_handler.addFilter(ExecutionContextFilter())
    process_handler.setFormatter(ConsoleFormatter(use_color=False))

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            console_handler,
            process_handler,
            ExecutionFileRouter(),
        ],
        force=True,
    )

    logging.getLogger("logging_setup").info(
        "Process logging ready. PID=%s; log file: %s",
        process_id,
        process_dir / "process.log",
    )

@contextmanager
def execution_logging(log_dir=None):
    """Create an isolated log directory for one execution."""
    run_id = uuid.uuid4().hex
    base_dir = Path(log_dir or os.getenv("LOG_DIR", "logs"))
    run_dir = base_dir / f"run-{os.getpid()}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)

    handler = RotatingFileHandler(
        run_dir / "execution.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=1,
        encoding="utf-8",
    )
    handler.setFormatter(ConsoleFormatter(use_color=False))
    state = ExecutionLogState(handler=handler)

    run_token = run_id_context.set(run_id)
    file_token = execution_log_context.set(state)
    logger = logging.getLogger("execution")

    try:
        logger.info("Execution started. Log directory: %s", run_dir)
        yield run_id
    except (asyncio.CancelledError, GeneratorExit):
        logger.info("Execution cancelled.")
        raise
    except BaseException:
        logger.exception("Execution failed.")
        raise
    else:
        logger.info("Execution scope finished.")
    finally:
        state.active = False
        execution_log_context.reset(file_token)
        run_id_context.reset(run_token)
        handler.close()

class ExecutionLoggingMiddleware:
    """Keep execution logging active for the whole HTTP response."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        with execution_logging():
            await self.app(scope, receive, send)
