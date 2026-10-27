"""Authenticated HTTP API and lease-based distributed workers."""
from __future__ import annotations

import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

from .jobs import JobConflict, JobQueue

KINDS = {"run", "generate", "edit", "evaluate"}


def terminate_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def check_configuration(kind, config, roots):
    if kind not in KINDS or not isinstance(config, dict):
        raise ValueError("job kind or configuration is invalid")
    roots = [Path(root).resolve() for root in roots]
    def check_path(value):
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError("service input paths must be absolute")
        path = Path(value).resolve()
        if not any(path.is_relative_to(root) for root in roots):
            raise ValueError("input path is outside configured artifact roots")
        if not path.exists():
            raise ValueError("input artifact does not exist")
    def walk(value, key=""):
        if isinstance(value, dict):
            for name, item in value.items():
                if name in ("output_dir", "artifact_root", "neural_output_dir"):
                    raise ValueError("workers own output directories")
                if name == "provider" and item != "https":
                    raise ValueError("serving supports HTTPS model providers")
                if name == "key_environment" and item != "HEARTHIS_AUDIO_API_KEY":
                    raise ValueError("provider credential environment is fixed by the server")
                if name == "endpoint" and item != os.environ.get("HEARTHIS_PROVIDER_ENDPOINT"):
                    raise ValueError("provider endpoint must match server configuration")
                walk(item, name)
        elif isinstance(value, list):
            for item in value:
                walk(item, key)
        elif key == "path" or key.endswith(("_path", "_manifest")) or key in ("recording_dir", "listening_correction", "waveforms", "source", "predicted", "reference", "spectrogram_manifests"):
            check_path(value)
    walk(config)
    required = {"generate": ("model_path", "music", "duration_seconds"), "edit": ("source", "model_path", "music", "edits"),
                "evaluate": (), "run": ()}[kind]
    if any(key not in config for key in required):
        raise ValueError("job configuration lacks required inputs")
    # Bound resource requests before queueing. Workers also enforce model contracts.
    bounds = {"iterations": 10000, "epochs": 10000, "draws": 16384, "particles": 1000000,
              "ideal_count": 256, "duration_seconds": 3600, "samples": 65536, "hidden": 4096,
              "latent_dimension": 512, "sweeps": 100000, "horizon_seconds": 86400}
    def limits(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in bounds and (isinstance(item, bool) or not isinstance(item, (float, int)) or not 0 < item <= bounds[key]):
                    raise ValueError(f"{key} exceeds serving resource limits")
                limits(item)
        elif isinstance(value, list):
            if len(value) > 10000:
                raise ValueError("job array exceeds serving limits")
            for item in value:
                limits(item)
    limits(config)


def execute_job(kind, configuration_path, output_dir):
    from .pipeline import run_configuration, write_json
    config = json.loads(Path(configuration_path).read_text())
    output_dir = Path(output_dir)
    if kind == "run":
        return run_configuration(Path(configuration_path), output_dir, device=os.environ.get("HEARTHIS_DEVICE", "cpu"))
    if kind == "evaluate":
        from .model.evaluation import evaluate_configuration
        return evaluate_configuration(configuration_path, output_dir)
    if kind == "edit":
        from .model.editing import edit_audio
        return edit_audio(output_dir=output_dir, **{**config, "source": Path(config["source"]), "model_path": Path(config["model_path"])})
    if kind == "generate":
        from .model.waveform import generate_ideal_audio
        if output_dir.exists():
            raise ValueError("generation requires a new output directory")
        options = {key: config[key] for key in ("duration_seconds", "seed", "sections") if key in config}
        generated = generate_ideal_audio({"samples": [{"music": config["music"]}]}, Path(config["model_path"]), output_dir / "audio", **options)
        if config.get("model_run_path"):
            from .model.evaluation import load_run
            from .audio.ingest import sha256_file
            _, _, _, _, joint = load_run(config["model_run_path"])
            if joint is None or Path(joint.model.waveform_path).resolve() != Path(config["model_path"]).resolve():
                raise ValueError("modality generation requires the matching joint model run")
            if joint.model.observation_kernel is not None:
                generated["generated_modalities"] = [joint.model.observation_kernel.sample(config["music"], seed=config.get("seed", 0))]
            generated["joint_model"] = {"path": str(Path(config["model_run_path"]).resolve()), "sha256": sha256_file(Path(config["model_run_path"]))}
        path = output_dir / "manifest.json"
        pending = output_dir / ".manifest.pending.json"
        write_json(pending, {"schema": "hearthis.generated-audio/v1", **generated})
        try:
            validate_result(pending)
            pending.replace(path)
        except Exception:
            pending.unlink(missing_ok=True)
            raise
        return path
    raise ValueError("unknown job kind")


def validate_result(path):
    from .model.validation import validate_end_to_end, validate_model_run
    from .audio.ingest import sha256_file
    data = json.loads(Path(path).read_text())
    schema = data["schema"]
    if schema == "hearthis.end-to-end/v1":
        return validate_end_to_end(Path(path))
    if schema == "hearthis.taste-program/v1":
        return validate_model_run(Path(path))
    if schema == "hearthis.audio-edit/v1":
        from .model.editing import validate_edit
        return validate_edit(path)
    if schema == "hearthis.evaluation/v1":
        from .model.evaluation import validate_evaluation
        return validate_evaluation(path)
    if schema == "hearthis.generated-audio/v1":
        from .model.waveform import WaveformModel
        from .audio.separation import _audio_geometry
        import numpy as np
        if sha256_file(Path(data["model"]["path"])) != data["model"]["sha256"]:
            raise ValueError("generation model hash mismatch")
        model = WaveformModel.load(Path(data["model"]["path"]))
        if data.get("joint_model"):
            reference = data["joint_model"]
            if sha256_file(Path(reference["path"])) != reference["sha256"]:
                raise ValueError("generation joint model hash mismatch")
            validate_model_run(Path(reference["path"]))
        if not data["waveforms"]:
            raise ValueError("generation lacks waveforms")
        for record in data["waveforms"]:
            if sha256_file(Path(record["path"])) != record["sha256"]:
                raise ValueError("generated audio hash mismatch")
            rate, audio = _audio_geometry(Path(record["path"]))
            if (rate != model.config["sample_rate_hz"] or audio.shape[1] != 1 or len(audio) != record["frame_count"]
                    or not np.isfinite(audio).all() or not np.isclose(len(audio) / rate, record["duration_seconds"])):
                raise ValueError("generated audio geometry mismatch")
        return {"valid": True, "waveforms": len(data["waveforms"])}
    raise ValueError("unknown result schema")


def worker(queue, root, *, stop=None, lease_seconds=120, poll_seconds=1, once=False):
    stop = stop or threading.Event()
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    health = Path(os.environ.get("HEARTHIS_WORKER_HEALTH_PATH", "/tmp/hearthis-worker-health"))
    while not stop.is_set():
        health.touch()
        job = queue.claim(lease_seconds=lease_seconds)
        if job is None:
            if once:
                return
            stop.wait(poll_seconds)
            continue
        identifier, token = str(job["id"]), str(job["lease_token"])
        attempt = root / identifier / f"attempt-{job['attempts']}-{token}"
        attempt.mkdir(parents=True, exist_ok=False)
        configuration = attempt / "configuration.json"
        from .pipeline import write_json
        write_json(configuration, job["configuration"])
        result = attempt / "result" / "manifest.json"
        log = attempt / "worker.log"
        command = [sys.executable, "-m", "hearthis.cli", "execute-job", job["kind"], str(configuration), "--output-dir", str(result.parent)]
        with log.open("wb") as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + int(os.environ.get("HEARTHIS_JOB_TIMEOUT_SECONDS", "7200"))
            lost = False
            while process.poll() is None:
                health.touch()
                if stop.wait(min(lease_seconds / 3, 5)) or time.monotonic() >= deadline:
                    terminate_group(process)
                    break
                try:
                    valid = queue.heartbeat(identifier, token, lease_seconds=lease_seconds)
                except Exception:
                    valid = False
                if not valid:
                    lost = True
                    terminate_group(process)
                    break
        if not lost:
            if process.returncode == 0 and result.exists():
                try:
                    validate_result(result)
                    queue.complete(identifier, token, result)
                except Exception:
                    queue.fail(identifier, token, "result validation failed", retryable=False)
            else:
                queue.fail(identifier, token, "worker execution failed; inspect attempt log", retryable=process.returncode != 2)
        if once:
            return


def artifact_index(result):
    """Only expose artifacts reachable from a validated completed manifest."""
    validate_result(result)
    from .audio.ingest import sha256_file
    index = {"manifest.json": Path(result)}
    data = json.loads(Path(result).read_text())
    if data["schema"] == "hearthis.end-to-end/v1":
        for i, reference in enumerate(data.get("edits", [])):
            path = Path(reference["path"])
            index[f"edits/{i}/manifest.json"] = path
            index[f"edits/{i}/audio.wav"] = Path(json.loads(path.read_text())["audio"]["path"])
        if data.get("evaluation"):
            index["evaluation.json"] = Path(data["evaluation"]["path"])
        model_path = Path(data["model"]["path"])
        index["model/manifest.json"] = model_path
        data = json.loads(model_path.read_text())
    for name, record in data.get("artifacts", {}).items():
        index[name] = Path(record["path"])
        if name == "catalog.json":
            for source in json.loads(index[name].read_text()).get("provenance", []):
                if source.get("recording_id") and source.get("references", {}).get("ingestion"):
                    index[f"sources/{source['recording_id']}/ingestion.json"] = Path(source["references"]["ingestion"]["path"])
        if name == "generated-audio.json":
            for i, wave in enumerate(json.loads(index[name].read_text())["waveforms"]):
                index[f"audio/{i}.wav"] = Path(wave["path"])
    for i, wave in enumerate(data.get("waveforms", [])):
        index[f"audio/{i}.wav"] = Path(wave["path"])
    if data["schema"] == "hearthis.audio-edit/v1":
        index["audio/edited.wav"] = Path(data["audio"]["path"])
    return {name: {"path": path, "sha256": sha256_file(path)} for name, path in index.items()}


def api_server(queue, roots, token, *, host="0.0.0.0", port=8080):
    if not token or len(token) < 24:
        raise ValueError("HEARTHIS_API_TOKEN must contain at least 24 characters")
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Do not log credentials, requests or personal feedback.

        def reply(self, status, data):
            encoded = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def authorized(self):
            return hmac.compare_digest(self.headers.get("Authorization", "").encode("utf-8"), ("Bearer " + token).encode("utf-8"))

        def do_GET(self):
            self.connection.settimeout(30)
            parsed = urlsplit(self.path)
            if parsed.path == "/healthz":
                return self.reply(200, {"healthy": True})
            if parsed.path == "/readyz":
                try:
                    queue.ready()
                    return self.reply(200, {"ready": True})
                except Exception:
                    return self.reply(503, {"ready": False})
            if not self.authorized():
                return self.reply(401, {"error": "authentication required"})
            parts = parsed.path.strip("/").split("/")
            if len(parts) not in (2, 3) or parts[0] != "jobs":
                return self.reply(404, {"error": "unknown endpoint"})
            try:
                job = queue.get(parts[1])
                if job is None:
                    return self.reply(404, {"error": "unknown job"})
                if len(parts) == 2:
                    return self.reply(200, job)
                if parts[2] != "artifacts" or job["state"] != "succeeded":
                    return self.reply(409, {"error": "job artifacts are not available"})
                files = artifact_index(Path(job["result_path"]))
                name = parse_qs(parsed.query).get("name", [None])[0]
                if name is None:
                    return self.reply(200, {"artifacts": {key: {"sha256": ref["sha256"], "size_bytes": ref["path"].stat().st_size} for key, ref in files.items()}})
                if name not in files:
                    return self.reply(404, {"error": "unknown artifact"})
                path = files[name]["path"]
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav" if path.suffix == ".wav" else "application/json")
                self.send_header("Content-Length", str(path.stat().st_size))
                self.send_header("ETag", '"' + files[name]["sha256"] + '"')
                self.end_headers()
                with path.open("rb") as stream:
                    while block := stream.read(65536):
                        self.wfile.write(block)
            except Exception:
                self.reply(503, {"error": "artifact or queue unavailable"})

        def do_POST(self):
            self.connection.settimeout(30)
            if not self.authorized():
                return self.reply(401, {"error": "authentication required"})
            parts = urlsplit(self.path).path.strip("/").split("/")
            feedback_endpoint = len(parts) == 3 and parts[0] == "jobs" and parts[2] == "feedback"
            if parts != ["jobs"] and not feedback_endpoint:
                return self.reply(404, {"error": "unknown endpoint"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1048576:
                    return self.reply(413, {"error": "request must contain at most 1 MiB of JSON"})
                request = json.loads(self.rfile.read(length), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
                if feedback_endpoint:
                    job = queue.get(parts[1])
                    if job is None or job["state"] != "succeeded" or job["kind"] != "run":
                        return self.reply(409, {"error": "feedback requires a completed model run"})
                    result = Path(job["result_path"])
                    validate_result(result)
                    outer = json.loads(result.read_text())
                    model = json.loads(Path(outer["model"]["path"]).read_text())
                    artifacts = model["artifacts"]
                    feedback = request.get("feedback", [])
                    if not feedback or any(not e.get("event_id") or not e.get("provenance") for e in feedback):
                        raise ValueError("served feedback requires event IDs and source provenance")
                    config = {"catalog_path": artifacts["catalog.json"]["path"],
                              "listener_path": artifacts["listener-posterior.json"]["path"],
                              "feedback": feedback, "intent": request.get("intent", {}),
                              "sampling": request.get("sampling", {"draws": 128, "particles": 512, "ideal_count": 1, "explain_count": 1})}
                    if "joint-posterior.json" in artifacts:
                        config["joint_posterior_path"] = artifacts["joint-posterior.json"]["path"]
                        config["joint"] = request.get("joint", {"iterations": 30, "draws": 16})
                    request = {"kind": "run", "configuration": config}
                check_configuration(request["kind"], request["configuration"], roots)
                job = queue.submit(request["kind"], request["configuration"], idempotency_key=self.headers.get("Idempotency-Key"))
                self.reply(202, job)
            except JobConflict as error:
                self.reply(409, {"error": str(error)})
            except (ValueError, KeyError, TypeError) as error:
                self.reply(400, {"error": str(error)})
            except Exception:
                self.reply(503, {"error": "queue unavailable"})
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["api", "worker", "init"])
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    queue = JobQueue(os.environ["HEARTHIS_DATABASE_URL"])
    queue.initialize()
    if args.role == "init":
        return
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    if args.role == "worker":
        worker(queue, os.environ.get("HEARTHIS_JOB_ROOT", "/data/jobs"), stop=stop,
               lease_seconds=int(os.environ.get("HEARTHIS_LEASE_SECONDS", "120")))
    else:
        roots = os.environ.get("HEARTHIS_INPUT_ROOTS", "/data").split(os.pathsep)
        server = api_server(queue, roots, os.environ["HEARTHIS_API_TOKEN"], port=args.port)
        server.timeout = 1
        while not stop.is_set():
            server.handle_request()
        server.server_close()


if __name__ == "__main__":
    main()
