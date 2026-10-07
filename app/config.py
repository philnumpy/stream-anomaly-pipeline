import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# protocol=2 because the native Windows build of Redis is version 5, which
# predates the newer RESP3 handshake the client would otherwise try first.
REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6380/0?protocol=2")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://anomaly@127.0.0.1:5433/anomaly")
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts"))

WINDOW = 50
N_FEATURES = 38

JOBS_STREAM = "jobs"
GROUP = "scorers"

# Backpressure: once this many scoring jobs are waiting, the API answers 503
# instead of letting the queue (and Redis memory) grow without bound.
MAX_BACKLOG = int(os.environ.get("MAX_BACKLOG", 20000))

# Fixed-window rate limit per client, counted in Redis so that all API
# replicas share one counter.
RATE_LIMIT_PER_SECOND = int(os.environ.get("RATE_LIMIT_PER_SECOND", 2000))

# A job that a worker read but has not acknowledged for this long is assumed
# to belong to a dead worker and is taken over by another one.
CLAIM_IDLE_MS = int(os.environ.get("CLAIM_IDLE_MS", 10000))
