"""Integration checks run against the deployment's actual PostgreSQL queue."""
import os
from concurrent.futures import ThreadPoolExecutor
import uuid

import pytest

pytest.importorskip("psycopg")
from hearthis.jobs import JobQueue, JobConflict


@pytest.fixture
def queue():
    dsn = os.environ.get("HEARTHIS_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("set HEARTHIS_TEST_DATABASE_URL for PostgreSQL queue integration")
    queue = JobQueue(dsn)
    queue.initialize()
    # This suite requires an isolated test database.
    with queue.connection() as connection:
        connection.execute("TRUNCATE hearthis_jobs")
    return queue


def test_idempotency_and_concurrent_claim_fencing(queue):
    key = str(uuid.uuid4())
    a = queue.submit("run", {"value": 1}, idempotency_key=key)
    b = queue.submit("run", {"value": 1}, idempotency_key=key)
    assert a["id"] == b["id"]
    with pytest.raises(JobConflict):
        queue.submit("run", {"value": 2}, idempotency_key=key)
    with ThreadPoolExecutor(4) as pool:
        claims = list(pool.map(lambda _: queue.claim(), range(4)))
    claimed = [row for row in claims if row]
    assert len(claimed) == 1
    job = claimed[0]
    assert queue.heartbeat(job["id"], job["lease_token"])
    assert not queue.complete(job["id"], str(uuid.uuid4()), "/wrong")
    assert queue.complete(job["id"], job["lease_token"], "/result")
    assert queue.get(a["id"])["state"] == "succeeded"


def test_expired_worker_cannot_complete_reclaimed_job(queue):
    a = queue.submit("run", {})
    first = queue.claim()
    with queue.connection() as connection:
        connection.execute("UPDATE hearthis_jobs SET lease_until=now()-interval '1 second' WHERE id=%s", (a["id"],))
    assert not queue.heartbeat(first["id"], first["lease_token"])
    second = queue.claim()
    assert second["attempts"] == 2
    assert not queue.complete(first["id"], first["lease_token"], "/stale")
    assert queue.complete(second["id"], second["lease_token"], "/current")


def test_failure_retries_and_exhaustion(queue):
    a = queue.submit("run", {}, max_attempts=2)
    first = queue.claim()
    assert queue.fail(first["id"], first["lease_token"], "transient")
    assert queue.get(a["id"])["state"] == "queued"
    second = queue.claim()
    assert queue.fail(second["id"], second["lease_token"], "permanent")
    assert queue.get(a["id"])["state"] == "failed"
    assert queue.claim() is None
