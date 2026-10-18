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
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

run_id_context = ContextVar("chabo_log_run_id", default="-")
execution_log_context = ContextVar("chabo_execution_log", default=None)
generation_log_context = ContextVar("chabo_generation_log", default=None)
_retention_lock = threading.Lock()
_last_retention_check = {}
_active_dirs_lock = threading.Lock()
_active_log_dirs = set()
_LOG_DIRECTORY_RE = re.compile(r"(?:run|process)-(\d+)-[0-9a-f]{32}")
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")
_RETENTION_SECONDS = 7 * 24 * 60 * 60
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"", "0", "false", "no", "off"}


@dataclass
class RequestLogState:
    """Per-request outcomes reported in the request summary.

    One mutable object per request, so updates from LangGraph's copied
    contexts and child tasks reach the middleware.
    """
    outcome: str = "not_started"
    retrieval_outcome: str = "not_started"
    documents: object = "-"
    blocked_by: str = "-"


def mark_generation_outcome(outcome):
    """Track generation status across tasks sharing one request context."""
    state = generation_log_context.get()
    if state is not None and state.outcome != "failed":
        state.outcome = outcome


def mark_retrieval_outcome(outcome, documents=0):
    """Record ok, empty, failed or rerank_fallback; a failure is kept."""
    state = generation_log_context.get()
    if state is not None and state.retrieval_outcome != "failed":
        state.retrieval_outcome = outcome
        state.documents = documents


def mark_blocked(by):
    """Record which guard blocked the request; the first block is kept."""
    state = generation_log_context.get()
    if state is not None and state.blocked_by == "-":
        state.blocked_by = by


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


class AccessLogQueryFilter(logging.Filter):
    """Remove query strings before access logs reach any handler."""

    def filter(self, record):
        if (
            record.name == "uvicorn.access"
            and isinstance(record.args, tuple)
            and len(record.args) == 5
        ):
            client, method, target, version, status = record.args
            if isinstance(target, str):
                target = target.partition("?")[0]
            record.args = (client, method, target, version, status)
        return True


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


def file_logging_enabled():
    """Return True when LOG_TO_FILE asks for log files (default: console only)."""
    value = os.getenv("LOG_TO_FILE", "false").strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError("LOG_TO_FILE must be true or false.")


def log_directory():
    """Return the log directory; an empty LOG_DIR means the default."""
    return Path(os.getenv("LOG_DIR", "").strip() or "logs")


def _open_file_handler(path, backup_count):
    """Return a rotating file handler for path; raise OSError on failure."""
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


def _create_file_handler(path, backup_count):
    """Return a file handler, or keep console logging if storage fails."""
    try:
        return _open_file_handler(path, backup_count)
    except OSError as e:
        logging.getLogger("logging_setup").warning(
            "Cannot create log file %s (%s: %s). "
            "Continuing with console logging.",
            path,
            type(e).__name__,
            e,
        )
        return None


def _mark_active(directory):
    with _active_dirs_lock:
        _active_log_dirs.add(str(directory.absolute()))


def _unmark_active(directory):
    with _active_dirs_lock:
        _active_log_dirs.discard(str(directory.absolute()))


def _is_active(directory):
    with _active_dirs_lock:
        return str(directory.absolute()) in _active_log_dirs


def _pid_running(pid):
    """Return True if pid exists in this PID namespace."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        # PermissionError and similar mean the process exists.
        return True
    return True


def _last_activity(directory):
    """Return the newest modification time of a log directory or its files."""
    newest = directory.stat().st_mtime
    for entry in directory.iterdir():
        if entry.is_file() and not entry.is_symlink():
            newest = max(newest, entry.stat().st_mtime)
    return newest


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
    """Remove request/process log directories older than seven days.

    Finished directories expire seven days after their `.finished` marker.
    Unfinished directories (left by a crashed or killed process) expire seven
    days after their last write, unless their process may still be running.
    """
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

        cutoff = time.time() - _RETENTION_SECONDS

        if not base_dir.is_dir():
            return

        for directory in base_dir.iterdir():
            match = _LOG_DIRECTORY_RE.fullmatch(directory.name)
            if match is None:
                continue
            if directory.is_symlink() or not directory.is_dir():
                continue
            if _is_active(directory):
                continue

            finished = directory / ".finished"

            try:
                if finished.is_symlink():
                    continue
                if finished.is_file():
                    reason = "completed"
                    last_time = finished.stat().st_mtime
                else:
                    # A matching PID other than ours may be a live process
                    # sharing this directory. Our own PID with an inactive
                    # directory is a previous process (e.g. container restart).
                    pid = int(match.group(1))
                    if pid != os.getpid() and _pid_running(pid):
                        continue
                    reason = "unfinished"
                    last_time = _last_activity(directory)

                if last_time >= cutoff:
                    continue

                shutil.rmtree(directory)
                logger.info(
                    "Removed %s log directory older than seven days: %s",
                    reason,
                    directory,
                )
            except FileNotFoundError:
                # Another process may already have removed the directory.
                continue
            except OSError:
                logger.warning(
                    "Cannot remove log directory: %s",
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
    finally:
        _unmark_active(process_dir)

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
    """Configure console logging first, then optional process-file logging.

    File logging is opt-in (LOG_TO_FILE=true). When it is enabled and the
    log directory is not writable, startup fails instead of continuing.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    if level_name not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError(
            "LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL."
        )
    log_level = getattr(logging, level_name)

    color_mode = os.getenv("LOG_COLOR", "auto").strip().lower()
    if color_mode not in {"auto", "always", "never"}:
        raise ValueError("LOG_COLOR must be auto, always, or never.")

    to_file = file_logging_enabled()

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
    access_logger = logging.getLogger("uvicorn.access")
    if not any(
        isinstance(log_filter, AccessLogQueryFilter)
        for log_filter in access_logger.filters
    ):
        access_logger.addFilter(AccessLogQueryFilter())

    # Keep routine HTTP traffic out of normal application logs.
    # DEBUG restores detailed HTTP diagnostics.
    http_log_level = (
        logging.DEBUG
        if log_level == logging.DEBUG
        else max(log_level, logging.WARNING)
    )
    for logger_name in ("httpx", "httpcore"):
        logging.getLogger(logger_name).setLevel(http_log_level)

    sys.excepthook = _log_uncaught_exception
    logger = logging.getLogger("logging_setup")

    if not to_file:
        logger.info(
            "File logging disabled (LOG_TO_FILE is not true); "
            "logging to console only."
        )
        return

    base_dir = log_directory()
    cleanup_completed_logs(base_dir)

    process_dir = base_dir / f"process-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        process_handler = _open_file_handler(
            process_dir / "process.log",
            backup_count=3,
        )
    except OSError as e:
        raise RuntimeError(
            f"LOG_TO_FILE=true but log directory {base_dir} is not writable "
            f"({type(e).__name__}: {e}). Fix LOG_DIR or its mount permissions, "
            "or unset LOG_TO_FILE to log to the console only."
        ) from e

    _mark_active(process_dir)
    logging.getLogger().addHandler(process_handler)
    atexit.register(
        _finish_process_logging,
        process_handler,
        process_dir,
    )
    logger.info(
        "Process logging ready. Log file: %s",
        process_dir / "process.log",
    )


def _bind_execution_context(run_id):
    return (
        run_id_context.set(run_id),
        execution_log_context.set(None),
        generation_log_context.set(RequestLogState()),
    )


def _reset_execution_context(tokens):
    run_token, file_token, generation_token = tokens
    generation_log_context.reset(generation_token)
    execution_log_context.reset(file_token)
    run_id_context.reset(run_token)


def _start_execution_log(execution_id, log_dir):
    """Create this execution's log file; return (handler, run_dir) or Nones."""
    if log_dir:
        base_dir = Path(log_dir)
    elif file_logging_enabled():
        base_dir = log_directory()
    else:
        return None, None

    cleanup_completed_logs(base_dir)
    run_dir = base_dir / f"run-{os.getpid()}-{execution_id}"
    handler = _create_file_handler(run_dir / "execution.log", backup_count=1)
    if handler is None:
        return None, None
    _mark_active(run_dir)
    return handler, run_dir


def _finish_execution_log(handler, run_dir):
    """Close an execution's log file and mark its directory as finished."""
    logger = logging.getLogger("execution")
    try:
        handler.close()
    except OSError:
        logger.warning("Could not close execution log file.", exc_info=True)
    else:
        try:
            (run_dir / ".finished").touch()
        except OSError:
            logger.warning(
                "Could not mark execution logs as finished.",
                exc_info=True,
            )
    finally:
        _unmark_active(run_dir)


def _activate_execution_log(handler, run_dir):
    """Route this context's records to handler; return its state."""
    logger = logging.getLogger("execution")
    if handler is None:
        logger.debug("Execution started without a dedicated log file.")
        return None
    state = ExecutionLogState(handler=handler)
    execution_log_context.set(state)
    logger.info("Execution started. Log directory: %s", run_dir)
    return state


@contextmanager
def execution_logging(log_dir=None, request_id=None):
    """Keep a request ID and, with file logging on, a unique log directory."""
    execution_id = uuid.uuid4().hex
    run_id = request_id or execution_id
    tokens = _bind_execution_context(run_id)
    handler = run_dir = state = None
    logger = logging.getLogger("execution")

    try:
        handler, run_dir = _start_execution_log(execution_id, log_dir)
        state = _activate_execution_log(handler, run_dir)
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
        _reset_execution_context(tokens)
        if handler is not None:
            _finish_execution_log(handler, run_dir)


@asynccontextmanager
async def aexecution_logging(log_dir=None, request_id=None):
    """Async execution_logging: file setup and retention run off the event loop."""
    execution_id = uuid.uuid4().hex
    run_id = request_id or execution_id
    tokens = _bind_execution_context(run_id)
    handler = run_dir = state = None
    logger = logging.getLogger("execution")

    try:
        setup = asyncio.ensure_future(
            asyncio.to_thread(_start_execution_log, execution_id, log_dir)
        )
        try:
            handler, run_dir = await asyncio.shield(setup)
        except asyncio.CancelledError:
            # Close the file the worker thread may still create.
            setup.add_done_callback(_close_abandoned_execution_log)
            raise
        state = _activate_execution_log(handler, run_dir)
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
        _reset_execution_context(tokens)
        # Closing is cheap and must not be interrupted by a second
        # cancellation, so it runs synchronously.
        if handler is not None:
            _finish_execution_log(handler, run_dir)


def _close_abandoned_execution_log(future):
    if future.cancelled() or future.exception() is not None:
        return
    handler, run_dir = future.result()
    if handler is not None:
        _finish_execution_log(handler, run_dir)


def _request_id_from_headers(scope):
    """Return a single valid X-Request-ID header value, else None."""
    values = [
        value
        for name, value in scope.get("headers", [])
        if name.lower() == b"x-request-id"
    ]
    if len(values) != 1:
        return None
    candidate = values[0].decode("ascii", errors="replace")
    return candidate if _REQUEST_ID_RE.fullmatch(candidate) else None


async def _send_internal_error(send):
    """Send the same plain-text 500 response Starlette would send."""
    body = b"Internal Server Error"
    await send({
        "type": "http.response.start",
        "status": 500,
        "headers": [
            (b"content-type", b"text/plain; charset=utf-8"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
    })
    await send({"type": "http.response.body", "body": body})


class ExecutionLoggingMiddleware:
    """Correlate request logs and keep them open through streaming."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("path", "").rstrip("/") == "/health"
        ):
            await self.app(scope, receive, send)
            return

        request_id = _request_id_from_headers(scope)

        async with aexecution_logging(request_id=request_id) as run_id:
            started = time.perf_counter()
            status = None
            response_complete = False
            client_disconnected = False
            outcome = "finished"
            logger = logging.getLogger("execution")

            async def receive_tracking_disconnect():
                nonlocal client_disconnected
                message = await receive()
                if message["type"] == "http.disconnect":
                    client_disconnected = True
                return message

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
                await self.app(
                    scope, receive_tracking_disconnect, send_with_request_id
                )
            except asyncio.CancelledError:
                outcome = "cancelled"
                raise
            except Exception:
                outcome = "failed"
                if status is not None:
                    # Too late for a 500; aexecution_logging logs it.
                    raise
                # Answer here, inside the request context, so the 500 keeps
                # its RUN id and x-request-id and is logged only once.
                logger.exception(
                    "Unhandled exception; returning HTTP 500."
                )
                await _send_internal_error(send_with_request_id)
            except BaseException:
                outcome = "failed"
                raise
            finally:
                if (
                    outcome == "finished"
                    and client_disconnected
                    and not response_complete
                ):
                    outcome = "client_disconnected"
                duration = time.perf_counter() - started
                request_state = (
                    generation_log_context.get() or RequestLogState()
                )

                if request_state.outcome == "failed":
                    logger.warning(
                        "Request ended with generation failure: "
                        "status=%s response_complete=%s",
                        status if status is not None else "-",
                        response_complete,
                    )
                if request_state.retrieval_outcome in (
                    "failed", "rerank_fallback"
                ):
                    logger.warning(
                        "Request ended with degraded retrieval: "
                        "retrieval_outcome=%s documents=%s status=%s",
                        request_state.retrieval_outcome,
                        request_state.documents,
                        status if status is not None else "-",
                    )

                logger.info(
                    "Request summary: method=%r path=%r status=%s "
                    "duration_s=%.3f response_complete=%s outcome=%s "
                    "generation_outcome=%s retrieval_outcome=%s "
                    "documents=%s blocked_by=%s",
                    scope.get("method", "-"),
                    scope.get("path", "-"),
                    status if status is not None else "-",
                    duration,
                    response_complete,
                    outcome,
                    request_state.outcome,
                    request_state.retrieval_outcome,
                    request_state.documents,
                    request_state.blocked_by,
                )
