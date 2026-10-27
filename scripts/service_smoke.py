"""Exercise actual HTTP jobs across Docker API, database and workers."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configuration", type=Path, help="rebased run configuration in the mounted catalog")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = dict(line.split("=", 1) for line in (root / ".env").read_text().splitlines() if line and not line.startswith("#"))
    base = f"http://127.0.0.1:{env.get('HEARTHIS_PORT', '8080')}"
    def request(path, payload=None, authenticated=True):
        headers = {"Authorization": "Bearer " + env["HEARTHIS_API_TOKEN"]} if authenticated else {}
        data = None if payload is None else json.dumps(payload).encode()
        if data is not None:
            headers["Content-Type"] = "application/json"
        with urllib.request.urlopen(urllib.request.Request(base + path, data=data, headers=headers), timeout=30) as response:
            return json.load(response)
    deadline = time.monotonic() + args.timeout
    while True:
        try:
            if request("/readyz", authenticated=False)["ready"]:
                break
        except (urllib.error.URLError, TimeoutError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("API did not become ready")
        time.sleep(1)
    try:
        request("/jobs/unknown", authenticated=False)
        raise RuntimeError("unauthenticated request was accepted")
    except urllib.error.HTTPError as error:
        assert error.code == 401
    def submit(kind, configuration):
        return request("/jobs", {"kind": kind, "configuration": configuration})["id"]
    def wait(identifier):
        while time.monotonic() < deadline:
            job = request("/jobs/" + identifier)
            if job["state"] == "succeeded":
                return job
            if job["state"] == "failed":
                raise RuntimeError(f"job {identifier} failed: {job['error']}")
            time.sleep(2)
        raise RuntimeError("jobs did not finish before timeout")
    config = json.loads(args.configuration.read_text())
    run = wait(submit("run", config))
    manifest = request(f"/jobs/{run['id']}/artifacts?name=manifest.json")
    catalog = request(f"/jobs/{run['id']}/artifacts?name=catalog.json")
    generated = request(f"/jobs/{run['id']}/artifacts?name=generated-audio.json")
    song = catalog["songs"][0]
    model_path = generated["model"]["path"]
    music = next(r["music"] for r in song["regions"] if r["kind"] == "global")
    targets = [min(1, value + 0.1) for value in music]
    ingestion = request(f"/jobs/{run['id']}/artifacts?name=sources/{song['song_id']}/ingestion.json")
    canonical = ingestion["canonical_audio"]["path"]
    submitted = [submit("generate", {"model_path": model_path, "music": music, "duration_seconds": 2}),
                 submit("edit", {"source": canonical, "model_path": model_path, "music": music,
                                 "edits": [{"start_seconds": 1, "end_seconds": 2, "music": targets}]}),
                 submit("evaluate", {"model_path": manifest["model"]["path"],
                                     "waveforms": [generated["waveforms"][0]["path"]],
                                     "sampling": {"chains": 4, "sweeps": 40, "draws": 32}})]
    completed = [run] + [wait(identifier) for identifier in submitted]
    for job in completed:
        index = request(f"/jobs/{job['id']}/artifacts")
        assert "manifest.json" in index["artifacts"]
    feedback = request(f"/jobs/{run['id']}/feedback", {"feedback": [
        {"event_id": "smoke-held-out-feedback-1", "kind": "behavior", "song_id": song["song_id"], "value": 1,
         "exposed": True, "propensity": 0.5, "provenance": "synthetic HTTP integration observation"}],
        "sampling": {"draws": 16, "particles": 64, "ideal_count": 1, "explain_count": 0},
        "joint": {"iterations": 12, "draws": 8}})
    completed.append(wait(feedback["id"]))
    workers = {job["worker_id"] for job in completed}
    if len(workers) < 2:
        raise RuntimeError("smoke run did not exercise multiple workers")
    print(json.dumps({"http_authentication": "passed", "jobs": [{"id": job["id"], "kind": job["kind"], "state": job["state"],
                           "worker_id": job["worker_id"], "result_path": job["result_path"]} for job in completed],
                      "distinct_workers": len(workers)}, indent=2))


if __name__ == "__main__":
    main()
