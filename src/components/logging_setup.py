"""Shared logging format for the Orchestrator."""

import logging
import os
import sys
import time
import asyncio
import uuid
import re
import shutil
import threading
import atexit
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

run_id_context = ContextVar("chabo_log_run_id", default="-")
execution_log_context = ContextVar("chabo_execution_log", default=None)
_retention_lock = threading.Lock()
_last_retention_check = {}
_LOG_DIRECTORY_RE = re.compile(r"(?:run|process)-\d+-[0-9a-f]{32}")
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")


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

def cleanup_completed_logs(base_dir):
    """Remove completed request/process log directories older than seven days."""
    logger = logging.getLogger("logging_setup")
    base_dir = Path(base_dir)
    key = str(base_dir.absolute())

    if not _retention_lock.acquire(blocking=False):
        return

    try:
        now = time.monotonic()
        previous = _last_retention_check.get(key)
        if previous is not None and now - previous < 60:
            return
        _last_retention_check[key] = now

        cutoff = time.time() - 7 * 24 * 60 * 60

        if not base_dir.is_dir():
            return

        for directory in base_dir.iterdir():
            if not _LOG_DIRECTORY_RE.fullmatch(directory.name):
                continue
            if directory.is_symlink() or not directory.is_dir():
                continue

            finished = directory / ".finished"

            try:
                if finished.is_symlink() or not finished.is_file():
                    continue
                if finished.stat().st_mtime >= cutoff:
                    continue

                shutil.rmtree(directory)
                logger.info(
                    "Removed completed log directory older than seven days: %s",
                    directory,
                )
            except FileNotFoundError:
                # Another process may already have removed the directory.
                continue
            except OSError:
                logger.warning(
                    "Cannot remove completed log directory: %s",
                    directory,
                    exc_info=True,
                )

    except OSError:
        logger.warning(
            "Cannot inspect execution logs for retention.",
            exc_info=True,
        )
    finally:
        _retention_lock.release()

def _finish_process_logging(handler, process_dir):
    """Close this process's log and mark its directory as completed."""
    logging.getLogger().removeHandler(handler)

    try:
        handler.close()
    except OSError:
        logging.getLogger("logging_setup").warning(
            "Could not close process log file.",
            exc_info=True,
        )
        return

    try:
        (process_dir / ".finished").touch()
    except FileNotFoundError:
        # The directory may already have been removed, e.g. by a test.
        return
    except OSError:
        logging.getLogger("logging_setup").warning(
            "Could not mark process logs as finished.",
            exc_info=True,
        )

def configure_logging():
    level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    if level_name not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError(
            "LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL."
        )
    log_level = getattr(logging, level_name)

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
        level=log_level,
        handlers=[console_handler, ExecutionFileRouter()],
        force=True,
    )
    cleanup_completed_logs(Path(os.getenv("LOG_DIR", "logs")))

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
        atexit.register(
            _finish_process_logging,
            process_handler,
            process_dir,
        )
        logging.getLogger("logging_setup").info(
            "Process logging ready. Log file: %s",
            process_dir / "process.log",
        )
    sys.excepthook = _log_uncaught_exception

@contextmanager
def execution_logging(log_dir=None, request_id=None):
    """Keep a request ID and a unique directory for each execution."""
    execution_id = uuid.uuid4().hex
    run_id = request_id or execution_id
    run_token = run_id_context.set(run_id)
    file_token = execution_log_context.set(None)
    handler = None
    state = None
    logger = logging.getLogger("execution")

    try:
        base_dir = Path(log_dir or os.getenv("LOG_DIR", "logs"))
        cleanup_completed_logs(base_dir)
        run_dir = base_dir / f"run-{os.getpid()}-{execution_id}"
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
            else:
                try:
                    (run_dir / ".finished").touch()
                except OSError:
                    logger.warning(
                        "Could not mark execution logs as finished.",
                        exc_info=True,
                    )

class ExecutionLoggingMiddleware:
    """Correlate request logs and keep them open through streaming."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") == "/health":
            await self.app(scope, receive, send)
            return

        request_id = None
        values = [
            value
            for name, value in scope.get("headers", [])
            if name.lower() == b"x-request-id"
        ]

        if len(values) == 1:
            candidate = values[0].decode("ascii", errors="replace")
            if _REQUEST_ID_RE.fullmatch(candidate):
                request_id = candidate

        with execution_logging(request_id=request_id) as run_id:
            started = time.perf_counter()
            status = None
            response_complete = False
            outcome = "finished"

            async def send_with_request_id(message):
                nonlocal status, response_complete

                if message["type"] == "http.response.start":
                    status = message["status"]
                    message = dict(message)
                    headers = [
                        (name, value)
                        for name, value in message.get("headers", [])
                        if name.lower() != b"x-request-id"
                    ]
                    headers.append(
                        (b"x-request-id", run_id.encode("ascii"))
                    )
                    message["headers"] = headers

                await send(message)

                if (
                    message["type"] == "http.response.body"
                    and not message.get("more_body", False)
                ):
                    response_complete = True

            try:
                await self.app(scope, receive, send_with_request_id)
            except asyncio.CancelledError:
                outcome = "cancelled"
                raise
            except BaseException:
                outcome = "failed"
                raise
            finally:
                duration = time.perf_counter() - started
                logging.getLogger("execution").info(
                    "Request summary: method=%r path=%r status=%s "
                    "duration_s=%.3f response_complete=%s outcome=%s",
                    scope.get("method", "-"),
                    scope.get("path", "-"),
                    status if status is not None else "-",
                    duration,
                    response_complete,
                    outcome,
                )


