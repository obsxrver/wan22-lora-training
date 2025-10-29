"""Flask application providing a drag-and-drop UI for WAN2.2 LoRA training."""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
)
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent.parent
WEBUI_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "webui_datasets"
MUSUBI_DIR = Path("/workspace/musubi-tuner")
HIGH_LOG = MUSUBI_DIR / "run_high.log"
LOW_LOG = MUSUBI_DIR / "run_low.log"

LOG_PATTERN = re.compile(r"steps:\s*\d+%.*?\|\s*(\d+)/\d+.*?avr_loss=([\d.]+)")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class NoiseMetrics:
    """Holds training metrics for either high or low noise runs."""

    current_step: Optional[int] = None
    current_loss: Optional[float] = None
    history: List[Dict[str, float]] = field(default_factory=list)

    def add_point(self, step: int, loss: float) -> None:
        self.current_step = step
        self.current_loss = loss
        self.history.append({"step": step, "loss": loss})


class TrainingManager:
    """Coordinator for running training processes and streaming updates."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._listeners: List[queue.Queue] = []
        self._process: Optional[subprocess.Popen[str]] = None
        self._stdout_buffer: List[str] = []
        self._status: str = "idle"
        self._dataset_dir: Optional[Path] = None
        self._dataset_config: Optional[Path] = None
        self._stop_event = threading.Event()
        self._watch_threads: List[threading.Thread] = []
        self._metrics: Dict[str, NoiseMetrics] = {
            "high": NoiseMetrics(),
            "low": NoiseMetrics(),
        }

    # ---------------------- listener helpers ----------------------
    def register_listener(self, event_queue: queue.Queue) -> None:
        with self._lock:
            self._listeners.append(event_queue)
            status_snapshot = self._status
            stdout_snapshot = list(self._stdout_buffer)[-200:]
            metrics_snapshot = {
                noise: metrics.history.copy()
                for noise, metrics in self._metrics.items()
                if metrics.history
            }

        event_queue.put({"type": "status", "status": status_snapshot})
        if stdout_snapshot:
            event_queue.put({"type": "log-batch", "lines": stdout_snapshot})
        for noise, history in metrics_snapshot.items():
            event_queue.put({"type": "metrics-batch", "noise": noise, "points": history})

    def unregister_listener(self, event_queue: queue.Queue) -> None:
        with self._lock:
            if event_queue in self._listeners:
                self._listeners.remove(event_queue)

    def _broadcast(self, payload: Dict) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            listener.put(payload)

    # ----------------------- status helpers -----------------------
    def _set_status(self, status: str) -> None:
        with self._lock:
            self._status = status
        self._broadcast({"type": "status", "status": status})

    def get_status(self) -> Dict:
        with self._lock:
            process_running = self._process is not None and self._process.poll() is None
            metrics = {
                noise: {
                    "current_step": data.current_step,
                    "current_loss": data.current_loss,
                    "history": list(data.history),
                }
                for noise, data in self._metrics.items()
            }
            dataset_path = str(self._dataset_config) if self._dataset_config else None
            stdout_tail = list(self._stdout_buffer)[-200:]
            status = self._status
        return {
            "status": status,
            "running": process_running,
            "dataset_config": dataset_path,
            "metrics": metrics,
            "logs": stdout_tail,
        }

    def _append_stdout(self, line: str) -> None:
        with self._lock:
            self._stdout_buffer.append(line)
            # prevent unbounded growth
            if len(self._stdout_buffer) > 2000:
                self._stdout_buffer = self._stdout_buffer[-2000:]
        self._broadcast({"type": "log", "line": line})

    def reset(self) -> None:
        with self._lock:
            self._stdout_buffer.clear()
            self._dataset_dir = None
            self._dataset_config = None
            self._metrics = {"high": NoiseMetrics(), "low": NoiseMetrics()}
        self._stop_event = threading.Event()
        self._watch_threads = []

    # ----------------------- dataset helpers ----------------------
    def _create_dataset(self, files: List, title_suffix: str) -> Path:
        ensure_directory(DATA_ROOT)
        dataset_dir = DATA_ROOT / f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{secure_filename(title_suffix) or 'dataset'}"
        ensure_directory(dataset_dir)

        for uploaded in files:
            if not uploaded.filename:
                continue
            filename = secure_filename(uploaded.filename)
            if not filename:
                continue
            destination = dataset_dir / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            uploaded.save(destination)

        # ensure caption + cache directories exist
        ensure_directory(dataset_dir / "cache")
        ensure_directory(dataset_dir / "videocache")

        dataset_config = dataset_dir / "dataset.toml"
        dataset_config.write_text(
            """[general]
resolution = [960, 960]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "{image_dir}"
cache_directory = "{cache_dir}"
num_repeats = 1

[[datasets]]
video_directory = "{video_dir}"
cache_directory = "{video_cache}"
frame_extraction = "full"
max_frames = 81
resolution = [298, 298]
""".format(
                image_dir=str(dataset_dir),
                cache_dir=str(dataset_dir / "cache"),
                video_dir=str(dataset_dir),
                video_cache=str(dataset_dir / "videocache"),
            )
        )

        with self._lock:
            self._dataset_dir = dataset_dir
            self._dataset_config = dataset_config

        return dataset_config

    # ----------------------- process helpers ----------------------
    def start_training(self, form: Dict[str, str], files: List) -> Dict:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("Training is already running")

        title_suffix = form.get("titleSuffix", "mylora").strip() or "mylora"
        author = form.get("author", "authorName").strip() or "authorName"
        save_every = form.get("saveEvery", "100").strip() or "100"
        cpu_threads = form.get("cpuThreads", "").strip()
        max_loader = form.get("maxLoaderWorkers", "").strip()
        upload_cloud = "Y" if form.get("uploadCloud") == "true" else "N"
        shutdown_instance = "Y" if form.get("shutdownInstance") == "true" else "N"

        dataset_config = self._create_dataset(files, title_suffix)

        # remove previous logs before starting new run
        for log_path in (HIGH_LOG, LOW_LOG):
            if log_path.exists():
                try:
                    log_path.unlink()
                except OSError:
                    pass

        env = os.environ.copy()
        env.update(
            {
                "WAN_TITLE_SUFFIX": title_suffix,
                "WAN_AUTHOR": author,
                "WAN_DATASET": str(dataset_config),
                "WAN_SAVE_EVERY": save_every,
                "WAN_UPLOAD_CLOUD": upload_cloud,
                "WAN_SHUTDOWN_INSTANCE": shutdown_instance,
                "WAN_PROCEED": "Y",
                "WAN_NON_INTERACTIVE": "1",
            }
        )
        if cpu_threads:
            env["WAN_CPU_THREADS_PER_PROCESS"] = cpu_threads
        if max_loader:
            env["WAN_MAX_DATA_LOADER_WORKERS"] = max_loader

        command = [str(BASE_DIR / "run_wan_training.sh")]

        self.reset()
        self._set_status("starting")

        process = subprocess.Popen(
            command,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        with self._lock:
            self._process = process

        self._stop_event.clear()
        self._start_watchers()

        threading.Thread(target=self._consume_stdout, args=(process,), daemon=True).start()
        threading.Thread(target=self._wait_for_completion, args=(process,), daemon=True).start()

        return {"dataset_config": str(dataset_config)}

    def _consume_stdout(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        first_line = True
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            if first_line:
                self._set_status("running")
                first_line = False
            self._append_stdout(line)
        process.stdout.close()

    def _wait_for_completion(self, process: subprocess.Popen[str]) -> None:
        return_code = process.wait()
        if return_code == 0:
            self._set_status("completed")
        else:
            self._set_status("failed")
        self._stop_event.set()

    def _start_watchers(self) -> None:
        for noise, path in (("high", HIGH_LOG), ("low", LOW_LOG)):
            thread = threading.Thread(
                target=self._watch_log,
                args=(path, noise),
                daemon=True,
            )
            thread.start()
            self._watch_threads.append(thread)

    def _watch_log(self, log_path: Path, noise: str) -> None:
        last_size = 0
        while not self._stop_event.is_set():
            self._read_log_updates(log_path, noise, last_size)
            if log_path.exists():
                try:
                    last_size = log_path.stat().st_size
                except OSError:
                    pass
            time.sleep(2)
        # one final read after stop to capture trailing lines
        self._read_log_updates(log_path, noise, last_size)

    def _read_log_updates(self, log_path: Path, noise: str, start_offset: int) -> None:
        if not log_path.exists():
            return
        try:
            with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
                if start_offset:
                    handle.seek(start_offset)
                for line in handle:
                    match = LOG_PATTERN.search(line)
                    if match:
                        step = int(match.group(1))
                        loss = float(match.group(2))
                        with self._lock:
                            metrics = self._metrics[noise]
                            metrics.add_point(step, loss)
                        self._broadcast(
                            {
                                "type": "metric",
                                "noise": noise,
                                "step": step,
                                "loss": loss,
                            }
                        )
        except OSError:
            return


training_manager = TrainingManager()
app = Flask(
    __name__,
    static_folder=str(WEBUI_DIR / "static"),
    template_folder=str(WEBUI_DIR / "templates"),
)


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/train", methods=["POST"])
def train() -> Response:
    if "files" not in request.files:
        return jsonify({"error": "Please provide at least one dataset file."}), 400

    files = request.files.getlist("files")
    if not any(file.filename for file in files):
        return jsonify({"error": "Please provide at least one dataset file."}), 400

    try:
        result = training_manager.start_training(request.form, files)
    except RuntimeError as exc:  # already running
        return jsonify({"error": str(exc)}), 409
    except Exception as exc:  # unexpected failure
        training_manager._set_status("failed")
        return jsonify({"error": f"Failed to start training: {exc}"}), 500

    return jsonify({"message": "Training started", **result})


@app.route("/status")
def status() -> Response:
    return jsonify(training_manager.get_status())


@app.route("/events")
def events() -> Response:
    def event_stream():
        q: queue.Queue = queue.Queue()
        training_manager.register_listener(q)
        try:
            while True:
                event = q.get()
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            training_manager.unregister_listener(q)

    return Response(event_stream(), mimetype="text/event-stream")


def main() -> None:
    ensure_directory(DATA_ROOT)
    app.run(host="0.0.0.0", port=int(os.environ.get("WAN_WEBUI_PORT", "7860")), threaded=True)


if __name__ == "__main__":
    main()
