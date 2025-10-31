import asyncio
import json
import os
import re
import secrets
import shutil
import subprocess
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = REPO_ROOT / "run_wan_training.sh"
INDEX_HTML_PATH = Path(__file__).with_name("index.html")
DATASET_ROOT = Path("/workspace/musubi-tuner/dataset")
LOG_DIR = Path("/workspace/musubi-tuner")
HIGH_LOG = LOG_DIR / "run_high.log"
LOW_LOG = LOG_DIR / "run_low.log"
API_KEY_CONFIG_PATH = Path.home() / ".config" / "vastai" / "vast_api_key"
MANAGE_KEYS_URL = "https://cloud.vast.ai/manage-keys"
CLOUD_SETTINGS_URL = "https://cloud.vast.ai/settings/"
DATASET_ROOT.mkdir(parents=True, exist_ok=True)

TOKEN_ENV_VAR = "JUPYTER_TOKEN"
AUTH_COOKIE_NAME = "token"
AUTH_QUERY_PARAM = "token"


def _load_auth_token() -> str:
    token = os.environ.get(TOKEN_ENV_VAR)
    if token:
        return token
    generated = secrets.token_hex(32)
    print(
        "[webui] JUPYTER_TOKEN environment variable not set. "
        "Generated temporary token for this process: %s" % generated
    )
    return generated


AUTH_TOKEN = _load_auth_token()


class TokenAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, token: str) -> None:
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next):
        if not self._token:
            return await call_next(request)

        cookie_token = request.cookies.get(AUTH_COOKIE_NAME)
        query_token = request.query_params.get(AUTH_QUERY_PARAM)
        token_source = None

        if cookie_token == self._token:
            token_source = "cookie"
        elif query_token == self._token:
            token_source = "query"

        if token_source is None:
            if request.method == "OPTIONS":
                return await call_next(request)
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)

        if token_source == "query" and request.method in {"GET", "HEAD"}:
            redirect_url = request.url.remove_query_params(AUTH_QUERY_PARAM)
            response = RedirectResponse(url=str(redirect_url), status_code=303)
            response.set_cookie(
                AUTH_COOKIE_NAME,
                self._token,
                httponly=True,
                secure=redirect_url.scheme == "https",
                samesite="lax",
                path="/",
            )
            return response

        response = await call_next(request)

        if token_source == "query" and cookie_token != self._token:
            response.set_cookie(
                AUTH_COOKIE_NAME,
                self._token,
                httponly=True,
                secure=request.url.scheme == "https",
                samesite="lax",
                path="/",
            )

        return response

STEP_PATTERNS = [
    re.compile(r"global_step(?:=|:)\s*(\d+)"),
    re.compile(r"step(?:=|:)\s*(\d+)"),
    re.compile(r"Iteration\s+(\d+)"),
    re.compile(r"steps:.*\|\s*(\d+)\s*/"),
]
EPOCH_PATTERNS = [
    re.compile(r"Epoch\s*\[(\d+)(?:/(\d+))?\]"),
    re.compile(r"Epoch\s+(\d+)(?:\s*/\s*(\d+))?"),
    re.compile(r"Epoch\s+(\d+):"),
    re.compile(r"epoch(?:=|:)\s*(\d+)(?:\s*/\s*(\d+))?"),
    re.compile(r"epoch\s+(\d+)(?:\s*/\s*(\d+))?", re.IGNORECASE),
]
LOSS_PATTERNS = [
    re.compile(r"train_loss(?:=|:)\s*([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)"),
    re.compile(r"loss(?:=|:)\s*([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)"),
    re.compile(r"Loss\s*=?\s*([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)"),
]
# we are parsing lines in /workspace/musubi-tuner/run_high.log and /workspace/musubi-tuner/run_low.log that look like this:
# steps:   1%|          | 30/5200 [01:38<4:43:19,  3.29s/it, avr_loss=0.129]
TOTAL_STEP_PATTERNS = [
    re.compile(r"steps:.*\|\s*\d+\s*/\s*(\d+)")
]
TIME_PATTERN = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?)<\s*(\d{1,2}:\d{2}(?::\d{2})?)")
MAX_HISTORY_POINTS = 2000
MAX_LOG_LINES = 400


def parse_cloud_connections(output: str) -> List[Dict[str, str]]:
    connections: List[Dict[str, str]] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("https://"):
            continue
        if stripped.lower().startswith("id"):
            continue
        parts = stripped.split()
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        connection_id = parts[0]
        cloud_type = parts[-1]
        name = " ".join(parts[1:-1]) if len(parts) > 2 else ""
        connections.append({"id": connection_id, "name": name, "cloud_type": cloud_type})
    return connections


def is_api_key_configured() -> bool:
    try:
        return API_KEY_CONFIG_PATH.exists() and API_KEY_CONFIG_PATH.read_text(encoding="utf-8").strip() != ""
    except OSError:
        return False


def maybe_set_container_api_key() -> None:
    container_key = os.environ.get("CONTAINER_API_KEY")
    if not container_key or is_api_key_configured():
        return
    if shutil.which("vastai") is None:
        return
    try:
        subprocess.run(
            ["vastai", "set", "api-key", container_key],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        return


async def gather_cloud_status() -> Dict[str, Any]:
    cli_available = shutil.which("vastai") is not None
    api_key_configured = is_api_key_configured()

    if not cli_available:
        return {
            "cli_available": False,
            "api_key_configured": api_key_configured,
            "permission_error": False,
            "has_connections": False,
            "can_upload": False,
            "message": "vastai CLI not found. Install it with: pip install vastai --user --break-system-packages",
            "connections": [],
        }

    try:
        process = await asyncio.create_subprocess_exec(
            "vastai",
            "show",
            "connections",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await process.communicate()
    except FileNotFoundError:
        return {
            "cli_available": False,
            "api_key_configured": api_key_configured,
            "permission_error": False,
            "has_connections": False,
            "can_upload": False,
            "message": "vastai CLI not found. Install it with: pip install vastai --user --break-system-packages",
            "connections": [],
        }

    output = (stdout_bytes + stderr_bytes).decode("utf-8", errors="ignore")
    lower_output = output.lower()
    permission_error = "failed with error 401" in lower_output
    connections: List[Dict[str, str]] = []

    if not permission_error:
        connections = parse_cloud_connections(output)

    has_connections = bool(connections)
    can_upload = cli_available and not permission_error and has_connections

    if permission_error:
        message = (
            "Current Vast.ai API key lacks the permissions required for cloud uploads. "
            f"Create a new key at {MANAGE_KEYS_URL} and save it below."
        )
    elif not has_connections:
        message = (
            "No cloud connections detected. Configure one at "
            f"{CLOUD_SETTINGS_URL} and open \"cloud connection\" to link storage."
        )
    elif process.returncode != 0:
        message = output.strip() or "Failed to query cloud connections."
    else:
        message = "Cloud uploads are ready to use."

    return {
        "cli_available": cli_available,
        "api_key_configured": api_key_configured,
        "permission_error": permission_error,
        "has_connections": has_connections,
        "can_upload": can_upload,
        "message": message,
        "connections": connections,
    }


maybe_set_container_api_key()

class TrainRequest(BaseModel):
    title_suffix: str = Field(default="mylora", min_length=1)
    author: str = Field(default="authorName", min_length=1)
    dataset_path: str = Field(default=str(DATASET_ROOT / "dataset.toml"))
    save_every: int = Field(default=100, ge=1)
    cpu_threads_per_process: Optional[int] = Field(default=None, ge=1)
    max_data_loader_workers: Optional[int] = Field(default=None, ge=1)
    upload_cloud: bool = True
    shutdown_instance: bool = True
    auto_confirm: bool = True


class ApiKeyRequest(BaseModel):
    api_key: str = Field(min_length=1)


class EventManager:
    def __init__(self) -> None:
        self._listeners: List[asyncio.Queue] = []
        self._lock = asyncio.Lock()

    async def register(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._listeners.append(queue)
        return queue

    async def unregister(self, queue: asyncio.Queue) -> None:
        async with self._lock:
            if queue in self._listeners:
                self._listeners.remove(queue)

    async def publish(self, event: Dict) -> None:
        async with self._lock:
            listeners = list(self._listeners)
        for queue in listeners:
            await queue.put(event)


class TrainingState:
    def __init__(self) -> None:
        self.process: Optional[asyncio.subprocess.Process] = None
        self.status: str = "idle"
        self.running: bool = False
        self.history: Dict[str, List[Dict[str, float]]] = {"high": [], "low": []}
        self.current: Dict[str, Optional[Dict[str, Any]]] = {"high": None, "low": None}
        self.pending: Dict[str, Dict[str, Optional[float]]] = {
            "high": {"step": None, "loss": None},
            "low": {"step": None, "loss": None},
        }
        self.logs: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self.stop_event: asyncio.Event = asyncio.Event()
        self.tasks: List[asyncio.Task] = []
        self.stop_requested: bool = False

    def reset_for_start(self) -> None:
        self.history = {"high": [], "low": []}
        self.current = {"high": None, "low": None}
        self.pending = {
            "high": {"step": None, "loss": None},
            "low": {"step": None, "loss": None},
        }
        self.logs.clear()
        self.stop_event = asyncio.Event()
        self.tasks = []
        self.stop_requested = False

    def mark_started(self, process: asyncio.subprocess.Process) -> None:
        self.reset_for_start()
        self.process = process
        self.status = "running"
        self.running = True

    def mark_finished(self, status: str) -> None:
        self.status = status
        self.running = False
        self.process = None
        self.stop_requested = False
        if not self.stop_event.is_set():
            self.stop_event.set()

    def snapshot(self) -> Dict:
        return {
            "status": self.status,
            "running": self.running,
            "high": {
                "history": list(self.history["high"]),
                "current": dict(self.current["high"]) if self.current["high"] else None,
            },
            "low": {
                "history": list(self.history["low"]),
                "current": dict(self.current["low"]) if self.current["low"] else None,
            },
            "logs": list(self.logs),
        }

    def add_task(self, task: asyncio.Task) -> None:
        self.tasks.append(task)

    async def wait_for_tasks(self) -> None:
        if not self.tasks:
            return
        done, pending = await asyncio.wait(self.tasks, timeout=0)
        for task in pending:
            task.cancel()

    def append_log(self, line: str) -> None:
        self.logs.append(line.rstrip())

    async def update_metrics(self, run: str, metrics: Dict[str, Optional[Any]]) -> Optional[Dict[str, Optional[Dict[str, Any]]]]:
        entry = self.pending[run]
        changed = False
        current = dict(self.current[run]) if self.current[run] else {}

        step_value = metrics.get("step")
        if step_value is not None:
            step_int = int(step_value)
            if entry["step"] != step_int:
                entry["step"] = step_int
            if current.get("step") != step_int:
                current["step"] = step_int
                changed = True

        loss_value = metrics.get("loss")
        if loss_value is not None:
            loss_float = float(loss_value)
            if entry["loss"] != loss_float:
                entry["loss"] = loss_float

        total_steps = metrics.get("total_steps")
        if total_steps is not None:
            total_int = int(total_steps)
            if current.get("total_steps") != total_int:
                current["total_steps"] = total_int
                changed = True

        epoch_value = metrics.get("epoch")
        if epoch_value is not None:
            epoch_int = int(epoch_value)
            # Only update epoch if new value is greater (prevent resets)
            current_epoch = current.get("epoch", 0)
            epoch_int = max(current_epoch, epoch_int)
            if current.get("epoch") != epoch_int:
                current["epoch"] = epoch_int
                changed = True

        total_epochs = metrics.get("total_epochs")
        if total_epochs is not None:
            total_epochs_int = int(total_epochs)
            if current.get("total_epochs") != total_epochs_int:
                current["total_epochs"] = total_epochs_int
                changed = True

        elapsed = metrics.get("time_elapsed")
        if elapsed is not None and current.get("time_elapsed") != elapsed:
            current["time_elapsed"] = str(elapsed)
            changed = True

        remaining = metrics.get("time_remaining")
        if remaining is not None and current.get("time_remaining") != remaining:
            current["time_remaining"] = str(remaining)
            changed = True

        point: Optional[Dict[str, Any]] = None
        history = self.history[run]
        if entry["step"] is not None and entry["loss"] is not None:
            point = {"step": int(entry["step"]), "loss": float(entry["loss"])}
            if current.get("step") != point["step"] or current.get("loss") != point["loss"]:
                changed = True
            current["step"] = point["step"]
            current["loss"] = point["loss"]
            if history and history[-1]["step"] == point["step"]:
                if history[-1].get("loss") != point["loss"]:
                    history[-1] = point
                    changed = True
            else:
                history.append(point)
                changed = True
                if len(history) > MAX_HISTORY_POINTS:
                    del history[: len(history) - MAX_HISTORY_POINTS]
            self.current[run] = current
            entry["loss"] = None
        else:
            if current:
                if not self.current[run] or current != self.current[run]:
                    changed = True
                self.current[run] = current

        if not changed and point is None:
            return None

        current_snapshot = dict(self.current[run]) if self.current[run] else None
        return {"point": point, "current": current_snapshot}


event_manager = EventManager()
training_state = TrainingState()
app = FastAPI(title="WAN 2.2 Training UI")

app.add_middleware(TokenAuthMiddleware, token=AUTH_TOKEN)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    if not INDEX_HTML_PATH.exists():
        raise HTTPException(status_code=500, detail="UI assets missing")
    return INDEX_HTML_PATH.read_text(encoding="utf-8")


def _clear_dataset_directory() -> None:
    try:
        if DATASET_ROOT.exists() and not DATASET_ROOT.is_dir():
            raise HTTPException(status_code=500, detail="Dataset path is not a directory")
        if not DATASET_ROOT.exists():
            DATASET_ROOT.mkdir(parents=True, exist_ok=True)
            return
        for entry in DATASET_ROOT.iterdir():
            try:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            except FileNotFoundError:
                continue
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to prepare dataset directory: {exc}") from exc
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)) -> Dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    _clear_dataset_directory()
    saved = []
    for file in files:
        filename = Path(file.filename).name
        if not filename:
            continue
        destination = DATASET_ROOT / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        await file.close()
        saved.append(str(destination))
    return {"saved": saved, "count": len(saved)}


@app.get("/cloud-status")
async def cloud_status() -> Dict[str, Any]:
    return await gather_cloud_status()


@app.post("/vast-api-key")
async def set_vast_api_key(payload: ApiKeyRequest) -> Dict[str, Any]:
    api_key = payload.api_key.strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required.")
    if shutil.which("vastai") is None:
        raise HTTPException(status_code=500, detail="vastai CLI is not installed on this instance.")

    env = os.environ.copy()
    try:
        process = await asyncio.create_subprocess_exec(
            "vastai",
            "set",
            "api-key",
            api_key,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout_bytes, stderr_bytes = await process.communicate()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="vastai CLI is not installed on this instance.") from exc

    message = (stdout_bytes + stderr_bytes).decode("utf-8", errors="ignore").strip()
    if process.returncode != 0:
        raise HTTPException(status_code=400, detail=message or "Failed to save API key.")

    status = await gather_cloud_status()
    return {"message": message or "API key saved.", "cloud_status": status}


async def stream_process_output(process: asyncio.subprocess.Process) -> None:
    if process.stdout is None:
        return
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        decoded = line.decode("utf-8", errors="ignore").rstrip()
        if decoded:
            training_state.append_log(decoded)
            await event_manager.publish({"type": "log", "line": decoded})


def parse_metrics(line: str) -> Dict[str, Optional[Any]]:
    step_value: Optional[int] = None
    loss_value: Optional[float] = None
    total_steps: Optional[int] = None
    epoch_value: Optional[int] = None
    total_epochs: Optional[int] = None
    elapsed_time: Optional[str] = None
    remaining_time: Optional[str] = None
    for pattern in STEP_PATTERNS:
        match = pattern.search(line)
        if match:
            try:
                step_value = int(match.group(1))
            except ValueError:
                step_value = None
            break
    for pattern in LOSS_PATTERNS:
        match = pattern.search(line)
        if match:
            try:
                loss_value = float(match.group(1))
            except ValueError:
                loss_value = None
            break
    for pattern in TOTAL_STEP_PATTERNS:
        match = pattern.search(line)
        if match:
            try:
                total_steps = int(match.group(1))
            except ValueError:
                total_steps = None
            break
    for pattern in EPOCH_PATTERNS:
        match = pattern.search(line)
        if match:
            try:
                epoch_value = int(match.group(1))
            except (ValueError, TypeError):
                epoch_value = None
            total_group: Optional[str] = None
            if match.lastindex and match.lastindex >= 2:
                total_group = match.group(2)
            if total_group is not None:
                try:
                    total_epochs = int(total_group)
                except ValueError:
                    total_epochs = None
            break
    time_match = TIME_PATTERN.search(line)
    if time_match:
        elapsed_time = time_match.group(1)
        remaining_time = time_match.group(2)
    return {
        "step": step_value,
        "loss": loss_value,
        "total_steps": total_steps,
        "epoch": epoch_value,
        "total_epochs": total_epochs,
        "time_elapsed": elapsed_time,
        "time_remaining": remaining_time,
    }


async def monitor_log_file(path: Path, run: str) -> None:
    position = 0
    while not training_state.stop_event.is_set():
        if path.exists():
            try:
                size = path.stat().st_size
                if size < position:
                    position = 0
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    handle.seek(position)
                    for line in handle:
                        metrics = parse_metrics(line)
                        result = await training_state.update_metrics(run, metrics)
                        if not result:
                            continue
                        event: Dict[str, Any] = {"type": "metrics", "run": run}
                        point = result.get("point")
                        current = result.get("current")
                        if point:
                            event.update({"step": point["step"], "loss": point["loss"]})
                        if current:
                            event["current"] = current
                        await event_manager.publish(event)
                    position = handle.tell()
            except OSError:
                position = 0
        await asyncio.sleep(1.0)
    # Flush remaining data after stop
    if path.exists():
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                handle.seek(position)
                for line in handle:
                    metrics = parse_metrics(line)
                    result = await training_state.update_metrics(run, metrics)
                    if not result:
                        continue
                    event: Dict[str, Any] = {"type": "metrics", "run": run}
                    point = result.get("point")
                    current = result.get("current")
                    if point:
                        event.update({"step": point["step"], "loss": point["loss"]})
                    if current:
                        event["current"] = current
                    await event_manager.publish(event)
        except OSError:
            pass


async def wait_for_completion(process: asyncio.subprocess.Process) -> None:
    returncode = await process.wait()
    if training_state.stop_requested:
        status = "stopped"
    else:
        status = "completed" if returncode == 0 else "failed"
    training_state.mark_finished(status)
    summary = f"Training {status} (return code {returncode})"
    training_state.append_log(summary)
    await event_manager.publish({"type": "log", "line": summary})
    await event_manager.publish({"type": "status", "status": status, "running": False, "returncode": returncode})


def build_command(payload: TrainRequest) -> List[str]:
    args = ["bash", str(RUN_SCRIPT)]
    args.extend(["--title-suffix", payload.title_suffix])
    args.extend(["--author", payload.author])
    args.extend(["--dataset", payload.dataset_path])
    args.extend(["--save-every", str(payload.save_every)])
    if payload.cpu_threads_per_process is not None:
        args.extend(["--cpu-threads-per-process", str(payload.cpu_threads_per_process)])
    if payload.max_data_loader_workers is not None:
        args.extend(["--max-data-loader-workers", str(payload.max_data_loader_workers)])
    args.extend(["--upload-cloud", "Y" if payload.upload_cloud else "N"])
    args.extend(["--shutdown-instance", "Y" if payload.shutdown_instance else "N"])
    if payload.auto_confirm:
        args.append("--auto-confirm")
    return args


@app.post("/train")
async def start_training(payload: TrainRequest) -> Dict:
    if not RUN_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="Training script not found")
    if training_state.running:
        raise HTTPException(status_code=409, detail="Training already in progress")

    for log_path in (HIGH_LOG, LOW_LOG):
        try:
            log_path.unlink()
        except FileNotFoundError:
            pass

    cloud_status = await gather_cloud_status()
    if payload.upload_cloud and not cloud_status.get("can_upload", False):
        payload.upload_cloud = False
        reason = cloud_status.get("message") or "Cloud uploads are not available."
        note = f"Cloud uploads disabled: {reason}"
        training_state.append_log(note)
        await event_manager.publish({"type": "log", "line": note})

    command = build_command(payload)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(REPO_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    training_state.mark_started(process)

    await event_manager.publish({"type": "status", "status": "running", "running": True})

    stdout_task = asyncio.create_task(stream_process_output(process))
    high_task = asyncio.create_task(monitor_log_file(HIGH_LOG, "high"))
    low_task = asyncio.create_task(monitor_log_file(LOW_LOG, "low"))
    wait_task = asyncio.create_task(wait_for_completion(process))
    training_state.add_task(stdout_task)
    training_state.add_task(high_task)
    training_state.add_task(low_task)
    training_state.add_task(wait_task)

    return {"status": "started"}


@app.post("/stop")
async def stop_training() -> Dict:
    if not training_state.running or training_state.process is None:
        raise HTTPException(status_code=409, detail="No training process to stop")

    process = training_state.process
    training_state.stop_requested = True
    training_state.status = "stopping"
    training_state.append_log("Stop requested by user. Attempting to terminate training process…")
    await event_manager.publish({"type": "log", "line": "Stop requested by user. Attempting to terminate training process…"})
    await event_manager.publish({"type": "status", "status": "stopping", "running": True})

    try:
        process.terminate()
    except ProcessLookupError:
        pass

    try:
        await asyncio.wait_for(process.wait(), timeout=15)
    except asyncio.TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            pass

    return {"status": "stopping"}


@app.get("/status")
async def status() -> Dict:
    return training_state.snapshot()


@app.get("/events")
async def events() -> StreamingResponse:
    queue = await event_manager.register()
    snapshot = training_state.snapshot()

    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'snapshot', **snapshot})}\n\n"
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            await event_manager.unregister(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await training_state.wait_for_tasks()
