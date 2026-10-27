"""Write Kubernetes Secrets locally without displaying credential values."""
import base64
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env = dict(line.split("=", 1) for line in (root / ".env").read_text().splitlines() if line and not line.startswith("#"))
encode = lambda value: base64.b64encode(value.encode()).decode()
password = env["HEARTHIS_DATABASE_PASSWORD"]
items = [
    {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": "runtime", "namespace": "hearthis"}, "type": "Opaque",
     "data": {"database-password": encode(password)}},
    {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": "runtime-env", "namespace": "hearthis"}, "type": "Opaque",
     "data": {"HEARTHIS_DATABASE_URL": encode(f"postgresql://hearthis:{password}@database/hearthis"),
              "HEARTHIS_API_TOKEN": encode(env["HEARTHIS_API_TOKEN"]),
              "HEARTHIS_AUDIO_API_KEY": encode(env.get("HEARTHIS_AUDIO_API_KEY", "")),
              "HEARTHIS_PROVIDER_ENDPOINT": encode(env.get("HEARTHIS_PROVIDER_ENDPOINT", ""))}},
]
path = root / "deploy" / "runtime-secrets.json"
with path.open("x") as stream:
    json.dump({"apiVersion": "v1", "kind": "List", "items": items}, stream, indent=2)
path.chmod(0o600)
print(path)
