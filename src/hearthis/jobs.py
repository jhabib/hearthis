"""Durable PostgreSQL queue with leases, idempotency and fenced completion."""
from __future__ import annotations

import hashlib
import json
import socket
import uuid


SCHEMA = """
CREATE TABLE IF NOT EXISTS hearthis_jobs (
 id uuid PRIMARY KEY, kind text NOT NULL, configuration jsonb NOT NULL,
 request_hash text NOT NULL, idempotency_key text UNIQUE,
 state text NOT NULL CHECK (state IN ('queued','running','succeeded','failed')),
 attempts integer NOT NULL DEFAULT 0, max_attempts integer NOT NULL,
 lease_token uuid, lease_until timestamptz, worker_id text, result_path text, error text,
 created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS hearthis_jobs_claim ON hearthis_jobs(state, created_at);
ALTER TABLE hearthis_jobs ADD COLUMN IF NOT EXISTS worker_id text;
"""


class JobConflict(ValueError):
    pass


class JobQueue:
    def __init__(self, dsn):
        import psycopg
        from psycopg.rows import dict_row
        self.psycopg, self.dict_row, self.dsn = psycopg, dict_row, dsn

    def connection(self):
        return self.psycopg.connect(self.dsn, row_factory=self.dict_row, connect_timeout=5)

    def initialize(self):
        with self.connection() as connection:
            # Multiple API/worker replicas may initialize concurrently.
            connection.execute("SELECT pg_advisory_xact_lock(74623819)")
            connection.execute(SCHEMA)

    def ready(self):
        with self.connection() as connection:
            connection.execute("SELECT 1 FROM hearthis_jobs LIMIT 1")
        return True

    def submit(self, kind, configuration, *, idempotency_key=None, max_attempts=3):
        if max_attempts not in range(1, 11) or (idempotency_key is not None and (not idempotency_key or len(idempotency_key) > 200)):
            raise ValueError("invalid retry count or idempotency key")
        serialized = json.dumps(configuration, sort_keys=True, allow_nan=False)
        request_hash = hashlib.sha256((kind + "\n" + serialized).encode()).hexdigest()
        identifier = str(uuid.uuid4())
        with self.connection() as connection:
            row = connection.execute("""INSERT INTO hearthis_jobs(id,kind,configuration,request_hash,idempotency_key,state,max_attempts)
                VALUES (%s,%s,%s::jsonb,%s,%s,'queued',%s) ON CONFLICT (idempotency_key) DO NOTHING RETURNING *""",
                (identifier, kind, serialized, request_hash, idempotency_key, max_attempts)).fetchone()
            if row is None:
                row = connection.execute("SELECT * FROM hearthis_jobs WHERE idempotency_key=%s", (idempotency_key,)).fetchone()
                if row["request_hash"] != request_hash:
                    raise JobConflict("idempotency key already belongs to another request")
        return public_job(row)

    def get(self, identifier):
        try:
            identifier = str(uuid.UUID(str(identifier)))
        except ValueError:
            return None
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM hearthis_jobs WHERE id=%s", (identifier,)).fetchone()
        return None if row is None else public_job(row)

    def claim(self, *, lease_seconds=120, worker_id=None):
        if not 2 <= lease_seconds <= 86400:
            raise ValueError("lease must be between two seconds and one day")
        token = str(uuid.uuid4())
        with self.connection() as connection:
            connection.execute("""UPDATE hearthis_jobs SET state='failed',error='retry limit exhausted after worker lease expired',updated_at=now()
                WHERE state='running' AND lease_until<=now() AND attempts>=max_attempts""")
            row = connection.execute("""UPDATE hearthis_jobs SET state='running',attempts=attempts+1,lease_token=%s,worker_id=%s,
                lease_until=now()+(%s * interval '1 second'),updated_at=now(),error=NULL
                WHERE id=(SELECT id FROM hearthis_jobs WHERE (state='queued' OR (state='running' AND lease_until<=now()))
                    AND attempts<max_attempts ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING *""",
                (token, worker_id or socket.gethostname(), lease_seconds)).fetchone()
        return row

    def heartbeat(self, identifier, token, *, lease_seconds=120):
        with self.connection() as connection:
            row = connection.execute("""UPDATE hearthis_jobs SET lease_until=now()+(%s * interval '1 second'),updated_at=now()
                WHERE id=%s AND lease_token=%s AND state='running' AND lease_until>now() RETURNING id""",
                (lease_seconds, identifier, token)).fetchone()
        return row is not None

    def complete(self, identifier, token, result_path):
        with self.connection() as connection:
            row = connection.execute("""UPDATE hearthis_jobs SET state='succeeded',result_path=%s,lease_until=NULL,updated_at=now()
                WHERE id=%s AND lease_token=%s AND state='running' AND lease_until>now() RETURNING id""",
                (str(result_path), identifier, token)).fetchone()
        return row is not None

    def fail(self, identifier, token, error, *, retryable=True):
        with self.connection() as connection:
            row = connection.execute("""UPDATE hearthis_jobs SET state=CASE WHEN %s AND attempts<max_attempts THEN 'queued' ELSE 'failed' END,
                error=%s,lease_until=NULL,updated_at=now() WHERE id=%s AND lease_token=%s AND state='running' AND lease_until>now() RETURNING id""",
                (retryable, error[:1000], identifier, token)).fetchone()
        return row is not None


def public_job(row):
    return {key: str(row[key]) if key in ("id", "created_at", "updated_at") else row[key]
            for key in ("id", "kind", "state", "attempts", "max_attempts", "worker_id", "result_path", "error", "created_at", "updated_at")}
