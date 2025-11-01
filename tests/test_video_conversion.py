import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_server_stubs() -> None:
    if "fastapi" not in sys.modules:
        fastapi_module = types.ModuleType("fastapi")

        class FakeFastAPI:
            def __init__(self, *args, **kwargs):
                pass

            def add_middleware(self, *args, **kwargs):
                pass

            def mount(self, *args, **kwargs):
                pass

            def get(self, *args, **kwargs):
                def decorator(func):
                    return func

                return decorator

            def post(self, *args, **kwargs):
                def decorator(func):
                    return func

                return decorator

            def delete(self, *args, **kwargs):
                def decorator(func):
                    return func

                return decorator

            def on_event(self, *args, **kwargs):
                def decorator(func):
                    return func

                return decorator

        class FakeHTTPException(Exception):
            def __init__(self, status_code: int = 500, detail: str | None = None):
                super().__init__(detail or "")
                self.status_code = status_code
                self.detail = detail or ""

        class FakeUploadFile:
            pass

        fastapi_module.FastAPI = FakeFastAPI
        fastapi_module.File = lambda *args, **kwargs: None
        fastapi_module.HTTPException = FakeHTTPException
        fastapi_module.Request = object
        fastapi_module.UploadFile = FakeUploadFile

        responses_module = types.ModuleType("fastapi.responses")

        class _HTMLResponse:
            def __init__(self, *args, **kwargs):
                pass

        class _RedirectResponse:
            def __init__(self, *args, **kwargs):
                pass

        class _StreamingResponse:
            def __init__(self, *args, **kwargs):
                pass

        responses_module.HTMLResponse = _HTMLResponse
        responses_module.RedirectResponse = _RedirectResponse
        responses_module.StreamingResponse = _StreamingResponse

        staticfiles_module = types.ModuleType("fastapi.staticfiles")

        class _StaticFiles:
            def __init__(self, *args, **kwargs):
                pass

        staticfiles_module.StaticFiles = _StaticFiles

        sys.modules["fastapi"] = fastapi_module
        sys.modules["fastapi.responses"] = responses_module
        sys.modules["fastapi.staticfiles"] = staticfiles_module
        fastapi_module.responses = responses_module
        fastapi_module.staticfiles = staticfiles_module

    if "pydantic" not in sys.modules:
        pydantic_module = types.ModuleType("pydantic")

        class _BaseModel:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        def _field(default=None, **kwargs):
            return default

        class _ValidationError(Exception):
            pass

        pydantic_module.BaseModel = _BaseModel
        pydantic_module.Field = _field
        pydantic_module.ValidationError = _ValidationError
        sys.modules["pydantic"] = pydantic_module

    if "starlette.middleware.base" not in sys.modules:
        starlette_module = types.ModuleType("starlette")
        middleware_base = types.ModuleType("starlette.middleware.base")

        class _BaseHTTPMiddleware:
            def __init__(self, app, *args, **kwargs):
                self.app = app

        middleware_base.BaseHTTPMiddleware = _BaseHTTPMiddleware
        sys.modules["starlette"] = starlette_module
        sys.modules["starlette.middleware.base"] = middleware_base

    if "starlette.responses" not in sys.modules:
        responses_module = types.ModuleType("starlette.responses")

        class _JSONResponse:
            def __init__(self, content=None, status_code=200):
                self.content = content
                self.status_code = status_code

        responses_module.JSONResponse = _JSONResponse
        sys.modules["starlette.responses"] = responses_module


_install_server_stubs()

import webui.server as server


DATASET_TOML_TEMPLATE = """
[[datasets]]
video_directory = "{video_dir}"
""".strip()


@pytest.fixture()
def dataset_with_video(tmp_path):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video_file = video_dir / "clip.mp4"
    video_file.write_bytes(b"dummy")
    dataset_config = tmp_path / "dataset.toml"
    dataset_config.write_text(DATASET_TOML_TEMPLATE.format(video_dir=str(video_dir)), encoding="utf-8")
    return dataset_config, video_file


def test_convert_dataset_videos_requires_ffmpeg(monkeypatch, dataset_with_video):
    dataset_config, _ = dataset_with_video

    monkeypatch.setattr(server.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError) as excinfo:
        server._convert_dataset_videos_to_16fps(dataset_config)

    assert "ffmpeg" in str(excinfo.value)


def test_convert_dataset_videos_logs_failures(monkeypatch, dataset_with_video):
    dataset_config, video_file = dataset_with_video

    monkeypatch.setattr(server.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    probe_calls = {
        "count": 0,
    }

    def fake_probe(path):
        assert path == video_file
        probe_calls["count"] += 1
        return {"fps": 30.0}

    monkeypatch.setattr(server, "_probe_video_metadata", fake_probe)

    def fake_run(command, check, stdout, stderr, text):
        assert command[0] == "/usr/bin/ffmpeg"
        if command[1:] == ["-version"]:
            return SimpleNamespace(returncode=0, stdout="ffmpeg version test", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="conversion failed")

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    summary = server._convert_dataset_videos_to_16fps(dataset_config)

    assert summary["total"] == 1
    assert summary["converted"] == 0
    assert summary["skipped"] == 0
    assert any("conversion failed" in error for error in summary["errors"])
    assert any("Using ffmpeg" in line for line in summary["logs"])
    assert probe_calls["count"] == 1


def test_convert_dataset_videos_success(monkeypatch, dataset_with_video):
    dataset_config, video_file = dataset_with_video

    monkeypatch.setattr(server.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    probe_values = [
        {"fps": 8.0},
        {"fps": 16.0},
    ]

    def fake_probe(path):
        assert path == video_file
        return probe_values.pop(0)

    monkeypatch.setattr(server, "_probe_video_metadata", fake_probe)

    def fake_run(command, check, stdout, stderr, text):
        if command[0] != "/usr/bin/ffmpeg":
            raise AssertionError("Unexpected command")
        if command[1:] == ["-version"]:
            return SimpleNamespace(returncode=0, stdout="ffmpeg version success", stderr="")
        output_path = Path(command[-1])
        output_path.write_bytes(b"converted")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    summary = server._convert_dataset_videos_to_16fps(dataset_config)

    assert summary["converted"] == 1
    assert summary["skipped"] == 0
    assert summary["errors"] == []
    assert any("Converted" in line for line in summary["logs"])
    assert any("Using ffmpeg" in line for line in summary["logs"])
    assert video_file.read_bytes() == b"converted"
