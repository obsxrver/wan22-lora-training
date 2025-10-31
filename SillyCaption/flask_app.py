from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

from flask import Blueprint, Flask, Response, abort, jsonify, render_template, request

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
TRAINING_DATASET_ROOT = Path("/workspace/musubi-tuner/dataset")
LIBRARY_ROOT = Path(__file__).resolve().parent / "Datasets"

bp = Blueprint(
    "sillycaption",
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
)


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _iter_dataset_files(root: Path) -> Iterable[Path]:
    for file_path in root.rglob("*"):
        if file_path.is_file():
            yield file_path


def _zip_directory(root: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in _iter_dataset_files(root):
            arcname = file_path.relative_to(root).as_posix()
            zf.write(file_path, arcname)
    buffer.seek(0)
    return buffer.read()


def _extract_zip_to_directory(archive: zipfile.ZipFile, destination: Path) -> None:
    if destination.exists():
        if destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive.extractall(destination)


@bp.route("/")
def index() -> str:
    return render_template("sillycaption/index.html")


@bp.route("/api/import/active")
def import_active() -> Response:
    _ensure_directory(TRAINING_DATASET_ROOT)
    if not any(TRAINING_DATASET_ROOT.iterdir()):
        return Response(status=204)
    data = _zip_directory(TRAINING_DATASET_ROOT)
    return Response(data, mimetype="application/zip")


@bp.route("/api/export/active", methods=["POST"])
def export_active() -> Response:
    file = request.files.get("archive")
    if file is None:
        abort(400, "archive file is required")
    try:
        with zipfile.ZipFile(file.stream) as zf:
            _extract_zip_to_directory(zf, TRAINING_DATASET_ROOT)
    except zipfile.BadZipFile as exc:
        abort(400, f"Invalid archive: {exc}")
    return Response(status=204)


@bp.route("/api/library/list")
def library_list() -> Response:
    _ensure_directory(LIBRARY_ROOT)
    names = [p.name for p in LIBRARY_ROOT.iterdir() if p.is_dir()]
    names.sort(key=str.lower)
    return jsonify({"datasets": names})


def _normalise_dataset_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        abort(400, "Dataset name is required")
    if "/" in cleaned or ".." in cleaned:
        abort(400, "Invalid dataset name")
    return cleaned


@bp.route("/api/library/save", methods=["POST"])
def library_save() -> Response:
    file = request.files.get("archive")
    name = request.form.get("name", "")
    if file is None:
        abort(400, "archive file is required")
    dataset_name = _normalise_dataset_name(name)
    target_dir = LIBRARY_ROOT / dataset_name
    _ensure_directory(LIBRARY_ROOT)
    try:
        with zipfile.ZipFile(file.stream) as zf:
            _extract_zip_to_directory(zf, target_dir)
    except zipfile.BadZipFile as exc:
        abort(400, f"Invalid archive: {exc}")
    return Response(status=204)


@bp.route("/api/library/load/<path:dataset>")
def library_load(dataset: str) -> Response:
    dataset_name = _normalise_dataset_name(dataset)
    target_dir = LIBRARY_ROOT / dataset_name
    if not target_dir.exists() or not target_dir.is_dir():
        abort(404, "Dataset not found")
    data = _zip_directory(target_dir)
    return Response(data, mimetype="application/zip")


def create_app() -> Flask:
    _ensure_directory(TRAINING_DATASET_ROOT)
    _ensure_directory(LIBRARY_ROOT)
    app = Flask(__name__)
    app.register_blueprint(bp)
    return app
