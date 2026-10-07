import psycopg

from .config import DATABASE_URL

SCHEMA = """
CREATE TABLE IF NOT EXISTS machines (
    machine_id  TEXT PRIMARY KEY,
    threshold   REAL NOT NULL
);

-- One row per scored reading: just the anomaly score, not the 38 raw metrics.
-- The primary key makes inserts idempotent, so a job that gets processed
-- twice after a worker crash cannot create a duplicate.
CREATE TABLE IF NOT EXISTS scores (
    machine_id  TEXT    NOT NULL REFERENCES machines(machine_id),
    ts          BIGINT  NOT NULL,
    score       REAL    NOT NULL,
    is_anomaly  BOOLEAN NOT NULL,
    latency_ms  INTEGER NOT NULL,   -- API received the reading -> score ready
    worker      TEXT    NOT NULL,
    PRIMARY KEY (machine_id, ts)
);

CREATE TABLE IF NOT EXISTS alerts (
    machine_id   TEXT   NOT NULL REFERENCES machines(machine_id),
    ts           BIGINT NOT NULL,
    score        REAL   NOT NULL,
    top_metrics  INTEGER[] NOT NULL,  -- the 3 metrics with the largest error
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (machine_id, ts)
);

CREATE INDEX IF NOT EXISTS alerts_created_at ON alerts (created_at DESC);
"""


def connect() -> psycopg.Connection:
    return psycopg.connect(DATABASE_URL)


def init_schema(thresholds: dict[str, float]) -> None:
    with connect() as conn:
        conn.execute(SCHEMA)
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO machines (machine_id, threshold) VALUES (%s, %s) "
                "ON CONFLICT (machine_id) DO UPDATE SET threshold = EXCLUDED.threshold",
                list(thresholds.items()),
            )
