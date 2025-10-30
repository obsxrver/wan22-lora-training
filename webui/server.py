import asyncio
import json
import os
import re
import shutil
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = REPO_ROOT / "run_wan_training.sh"
INDEX_HTML_PATH = Path(__file__).with_name("index.html")
DATASET_ROOT = Path("/workspace/musubi-tuner/dataset")
LOG_DIR = Path("/workspace/musubi-tuner")
HIGH_LOG = LOG_DIR / "run_high.log"
LOW_LOG = LOG_DIR / "run_low.log"
DATASET_ROOT.mkdir(parents=True, exist_ok=True)

STEP_PATTERNS = [
    re.compile(r"global_step(?:=|:)\s*(\d+)"),
    re.compile(r"step(?:=|:)\s*(\d+)"),
    re.compile(r"Iteration\s+(\d+)"),
    re.compile(r"steps:.*\|\s*(\d+)\s*/"),
]
LOSS_PATTERNS = [
    re.compile(r"train_loss(?:=|:)\s*([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)"),
    re.compile(r"loss(?:=|:)\s*([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)"),
    re.compile(r"Loss\s*=?\s*([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)"),
]
# we are parsing lines in /workspace/musubi-tuner/ruh_high.log and /workspace/musubi-tuner/run_low.log that look like this:
# steps:   1%|          | 30/5200 [01:38<4:43:19,  3.29s/it, avr_loss=0.129]
MAX_HISTORY_POINTS = 2000
MAX_LOG_LINES = 400


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
        self.current: Dict[str, Optional[Dict[str, float]]] = {"high": None, "low": None}
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
                "current": self.current["high"],
            },
            "low": {
                "history": list(self.history["low"]),
                "current": self.current["low"],
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

    async def update_metrics(self, run: str, step: Optional[int], loss: Optional[float]) -> Optional[Dict[str, float]]:
        entry = self.pending[run]
        if step is not None:
            entry["step"] = int(step)
        if loss is not None:
            entry["loss"] = float(loss)
        if entry["step"] is None or entry["loss"] is None:
            return None
        point = {"step": int(entry["step"]), "loss": float(entry["loss"])}
        history = self.history[run]
        if history and history[-1]["step"] == point["step"]:
            history[-1] = point
        else:
            history.append(point)
            if len(history) > MAX_HISTORY_POINTS:
                del history[: len(history) - MAX_HISTORY_POINTS]
        self.current[run] = point
        entry["loss"] = None
        return point


event_manager = EventManager()
training_state = TrainingState()
app = FastAPI(title="WAN 2.2 Training UI")


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


def parse_metrics(line: str) -> Dict[str, Optional[float]]:
    step_value: Optional[int] = None
    loss_value: Optional[float] = None
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
    return {"step": step_value, "loss": loss_value}


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
                        if metrics["step"] is None and metrics["loss"] is None:
                            continue
                        point = await training_state.update_metrics(run, metrics["step"], metrics["loss"])
                        if point:
                            await event_manager.publish(
                                {"type": "metrics", "run": run, "step": point["step"], "loss": point["loss"]}
                            )
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
                    if metrics["step"] is None and metrics["loss"] is None:
                        continue
                    point = await training_state.update_metrics(run, metrics["step"], metrics["loss"])
                    if point:
                        await event_manager.publish(
                            {"type": "metrics", "run": run, "step": point["step"], "loss": point["loss"]}
                        )
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
