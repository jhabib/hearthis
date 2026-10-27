# Distributed serving

The HTTP API queues immutable configurations in PostgreSQL. Independent workers
claim jobs with `FOR UPDATE SKIP LOCKED`, renew leases, and execute isolated
subprocesses. Completion is fenced by the current lease token and expiry.
Every retry writes a new attempt directory. A worker verifies the result before
publishing its path. API artifact downloads revalidate the completed result.

Supported jobs are `run`, `generate`, `edit`, and `evaluate`. Model runs include
audio ingestion/analysis, optional HTTPS audio-LLM review and semantic grounding,
joint fitting, recommendations, playlists, explanations, generation, editing,
and evaluation. Cached validated recordings can skip preprocessing.

## Local Docker environment

From the repository root:

```sh
.venv/bin/python scripts/local_environment.py
.venv/bin/python -m hearthis.cli bundle examples/run-complete.json \
  --source-root . --output-dir artifacts/container-catalog-v3 \
  --target-prefix /catalog
docker compose up -d --build --scale worker=2
.venv/bin/python scripts/service_smoke.py \
  artifacts/container-catalog-v3/examples/run-complete.json
```

The bundle command requires the local FMA recordings referenced by the example.
It copies the required artifact graph, rebases paths, verifies original hashes,
and propagates new JSON hashes. Choose a new bundle directory on subsequent
exports and set `HEARTHIS_CATALOG_DIRECTORY` in `.env` to that directory.
The API listens on localhost port 18080 when using `local_environment.py`.
Credentials are generated into a mode-0600 ignored `.env` file; the helper
preserves an existing file. Set `HEARTHIS_CATALOG_DIRECTORY` to
`./artifacts/container-catalog-v3` for the commands above.

The Compose environment includes PostgreSQL storage and shared artifact storage
in named volumes. API and worker containers run as UID 1000. Worker limits are
two CPUs and 4 GiB of RAM, with a configurable two-hour job deadline. Database,
API and worker health checks are included. API requests use a bearer token,
accept at most 1 MiB of JSON, validate input roots and resource bounds, and
support `Idempotency-Key`. The provider endpoint and credential environment
are fixed by server configuration. The database is not exposed on the host.

```sh
docker compose ps
docker compose logs --tail=100 worker
docker compose down
```

Stopping the environment preserves its data volumes. Back up PostgreSQL and
the artifact volume together to preserve queue/result lineage. A terminated
worker ends its subprocess group; unfinished jobs can retry after lease expiry.
Permanently invalid configurations fail without repeated attempts.

## HTTP contract

All endpoints except `/healthz` and `/readyz` require
`Authorization: Bearer <HEARTHIS_API_TOKEN>`.

| Request | Result |
| --- | --- |
| `POST /jobs` with `{kind, configuration}` | HTTP 202 and job ID; equal idempotency requests reuse the ID, conflicting content returns 409 |
| `GET /jobs/{id}` | State, attempts, worker ID, result path and sanitized failure message |
| `GET /jobs/{id}/artifacts` | Names, sizes and hashes of verified completed artifacts |
| `GET /jobs/{id}/artifacts?name=audio/0.wav` | Streamed WAV with content hash ETag |
| `POST /jobs/{id}/feedback` | Queue a new version using that run's persistent listener/catalog/joint posterior |

Feedback requests contain `feedback`, optional `intent`, `sampling`, and
`joint`. Each feedback event needs a unique `event_id` and source `provenance`,
as well as the model's explicit exposure/propensity/outcome fields. The saved
listener maintains an event ledger that rejects repeated observations. The
parent run is immutable; clients choose the resulting job ID as their next
state version. Concurrent requests against the same parent create explicit
branches. There is no implicit mutable "latest listener" pointer.

All service input paths must be absolute and inside configured artifact roots.
Generated outputs belong to workers. Raw sources must already be mounted under
an allowed root; the API does not expose an unrestricted filesystem upload.
HTTPS model calls require an audio-capable provider configured by the operator.
Meta CLI transport remains available in the local CLI environment.

## Kubernetes environment

`deploy/kubernetes.yaml` contains a namespace, PostgreSQL StatefulSet, API and
worker Deployments, Services, resource bounds, probes, and storage claims.
It uses a ReadWriteMany artifact PVC so workers on different nodes see the
same inputs/results. Select a cluster storage class providing that capability
and use an image tag available to the cluster. PostgreSQL uses its own
ReadWriteOnce volume.

Generate the ignored Secrets file locally, then apply it to your chosen cluster:

```sh
.venv/bin/python deploy/kubernetes_secrets.py
kubectl create namespace hearthis
kubectl apply -f deploy/runtime-secrets.json
kubectl apply -f deploy/kubernetes.yaml
kubectl -n hearthis rollout status deployment/api
kubectl -n hearthis rollout status deployment/worker
kubectl -n hearthis port-forward service/api 18080:8080
```

Export a bundle with `--target-prefix /data/catalog` and place it on the
artifact PVC at `/data/catalog`. Kubernetes Services are internal by default.
Configure TLS at your ingress before exposing the API beyond a trusted local
connection. Database availability, backups and cluster storage are operator
responsibilities. The local Docker execution and Kubernetes configuration
validation are distinct verification scopes.
