# End-to-end completion contract

Implement executable paths for broad music/lyric/metadata observations, joint
neural and variational learning, temporally dependent waveform generation,
localized editing, held-out evaluation and sampler diagnostics, portable
artifacts, and durable distributed HTTP serving.

Compatibility: existing catalogs, decoder checkpoints, commands, and manifests
remain readable. Missing semantic observations remain uncertain priors. LLM
outputs retain AI provenance. Persistent listener state and session intent have
separate ownership. Each processing attempt owns a fresh directory. Only
validated outputs become successful runs or completed jobs.

Proof: focused mathematical and failure-path tests, full regression suite,
actual audio/model runs, independent manifest verification, Docker image build,
and HTTP requests executed through API, PostgreSQL, and multiple workers.
Deployment files must include health checks, shared artifact storage, queue
leases, retries, authentication, resource bounds, and shutdown behavior.

Empirical accuracy and musical quality require measured results from an
independent corpus; implementing evaluation does not supply those results.

Completed proof: full local run and independent verification; a distinct third
recording for waveform evaluation; live Docker API, PostgreSQL and two workers
executing run/generate/edit/evaluate/feedback jobs; current container source
hashes matching all 34 Python source files; isolated PostgreSQL concurrency,
lease, retry and idempotency tests; and eight valid Kubernetes schema resources.
Detailed results are in `COMPLETION.md` and ignored verification artifacts.

New uploads for broader Gemini annotations remain pending the destination
authorization requested after automatic approval review rejected that action.
The completed runs use existing live LLM reviews. Semantic-review transport and
abstention are tested with explicitly labeled fixtures.
