"""Shared authenticated Chat-Completions transport for listening and grounding."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import urlsplit
import urllib.error
import urllib.request


def complete(payload, *, provider="https", endpoint=None, key_environment="HEARTHIS_AUDIO_API_KEY",
             timeout=180, transport=None, purpose="hearthis music model"):
    if transport is not None:
        return transport(payload), "test-transport"
    request_bytes = json.dumps(payload).encode("utf-8")
    if provider == "meta-ape":
        annotation = subprocess.run(["meta", "metacli.command", "annotations", "--command", "ape.model raw", "-o", "json"],
                                    check=True, capture_output=True, text=True, timeout=30)
        if json.loads(annotation.stdout).get("control") != "AUTONOMOUS":
            raise RuntimeError("Meta CLI requires user confirmation for this model call")
        with tempfile.TemporaryDirectory(prefix="hearthis-request-") as temporary:
            body = Path(temporary) / "request.json"
            body.write_bytes(request_bytes)
            response = subprocess.run(["meta", "ape.model", "raw", "--body", f"file://{body}", "--purpose", purpose, "-o", "json"],
                                      capture_output=True, text=True, timeout=timeout)
            if response.returncode:
                raise RuntimeError(f"Meta APE model request failed with exit code {response.returncode}")
            return json.loads(response.stdout), "meta-ape/ape.model.raw"
    if provider != "https":
        raise ValueError("unknown model provider")
    endpoint = endpoint or os.environ.get("HEARTHIS_AUDIO_ENDPOINT")
    key = os.environ.get(key_environment)
    if not endpoint or not key:
        raise RuntimeError("configure HEARTHIS_AUDIO_ENDPOINT and the requested key environment variable")
    parsed = urlsplit(endpoint)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("model endpoint must be an authenticated HTTPS URL")
    request = urllib.request.Request(endpoint, data=request_bytes, method="POST",
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response), endpoint
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"model provider returned HTTP {error.code}") from None
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"model provider connection failed: {type(error).__name__}") from None


def response_json(response):
    content = response["choices"][0]["message"]["content"].strip()
    if content.startswith("```") and content.endswith("```"):
        content = "\n".join(content.splitlines()[1:-1])
    return json.loads(content)
