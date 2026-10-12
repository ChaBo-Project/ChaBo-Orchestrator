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

def _create_file_handler(path, backup_count):
    """Return a file handler, or keep console logging if storage fails."""
    handler = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path,
            maxBytes=5 * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.addFilter(ExecutionContextFilter())
        handler.setFormatter(ConsoleFormatter(use_color=False))
        return handler
    except OSError:
        if handler is not None:
            handler.close()
        logging.getLogger("logging_setup").warning(
            "Cannot create log file %s. Continuing with console logging.",
            path,
            exc_info=True,
        )
        return None

def _log_uncaught_exception(exc_type, exc_value, traceback):
    """Record uncaught main-thread exceptions through configured logging."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, traceback)
        return

    logging.getLogger("execution").critical(
        "Uncaught exception; process terminating.",
        exc_info=(exc_type, exc_value, traceback),
    )

def configure_logging():
    """Configure console logging first, then optional process-file logging."""
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

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, ExecutionFileRouter()],
        force=True,
    )

    process_dir = (
        Path(os.getenv("LOG_DIR", "logs"))
        / f"process-{os.getpid()}-{uuid.uuid4().hex}"
    )
    process_handler = _create_file_handler(
        process_dir / "process.log",
        backup_count=3,
    )

    if process_handler is not None:
        logging.getLogger().addHandler(process_handler)
        logging.getLogger("logging_setup").info(
            "Process logging ready. Log file: %s",
            process_dir / "process.log",
        )
    sys.excepthook = _log_uncaught_exception

@contextmanager
def execution_logging(log_dir=None):
    """Keep an execution ID even when its log file cannot be created."""
    run_id = uuid.uuid4().hex
    run_token = run_id_context.set(run_id)
    file_token = execution_log_context.set(None)
    handler = None
    state = None
    logger = logging.getLogger("execution")

    try:
        base_dir = Path(log_dir or os.getenv("LOG_DIR", "logs"))
        run_dir = base_dir / f"run-{os.getpid()}-{run_id}"
        handler = _create_file_handler(
            run_dir / "execution.log",
            backup_count=1,
        )

        if handler is not None:
            state = ExecutionLogState(handler=handler)
            execution_log_context.set(state)
            logger.info("Execution started. Log directory: %s", run_dir)
        else:
            logger.info("Execution started without a dedicated log file.")

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
        if state is not None:
            state.active = False
        execution_log_context.reset(file_token)
        run_id_context.reset(run_token)
        if handler is not None:
            try:
                handler.close()
            except OSError:
                logger.warning(
                    "Could not close execution log file.",
                    exc_info=True,
                )

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
