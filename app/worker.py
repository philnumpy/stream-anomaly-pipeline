"""Stream worker. Reads scoring jobs from the Redis stream as one member of a
consumer group, runs the LSTM on them in batches and stores the results.

Delivery is at-least-once: a job is acknowledged only after its result is
committed to PostgreSQL. If this process dies in between, the job stays in the
group's pending list and another worker takes it over. Because the result
tables are keyed on (machine_id, ts), doing a job twice changes nothing.
"""

import json
import os
import signal
import socket
import time
from collections import defaultdict

import numpy as np
import onnxruntime as ort
import redis

from . import db
from .config import (
    ARTIFACTS_DIR,
    CLAIM_IDLE_MS,
    GROUP,
    JOBS_STREAM,
    N_FEATURES,
    REDIS_URL,
    WINDOW,
)

# Must be unique per worker: the consumer group tells workers apart by name.
# In containers every worker has the same PID, so the hostname is included.
NAME = os.environ.get("WORKER_NAME", f"worker-{socket.gethostname()}-{os.getpid()}")
BATCH = 128


class Scorer:
    """Loads each machine's ONNX model the first time that machine shows up."""

    def __init__(self):
        self.thresholds = json.loads((ARTIFACTS_DIR / "thresholds.json").read_text())
        self.sessions: dict[str, ort.InferenceSession] = {}
        self.options = ort.SessionOptions()
        self.options.intra_op_num_threads = 1

    def score(self, machine: str, windows: np.ndarray):
        if machine not in self.sessions:
            self.sessions[machine] = ort.InferenceSession(
                str(ARTIFACTS_DIR / f"{machine}.onnx"), self.options
            )
        # Same preprocessing as training (anomaly.model.preprocess).
        windows = np.clip(windows, -1.0, 2.0)
        (recon,) = self.sessions[machine].run(None, {"window": windows})
        errors = (recon - windows[:, -1, :]) ** 2
        scores = errors.mean(axis=1)
        return scores, scores > self.thresholds[machine], errors


def process(entries, scorer: Scorer, conn, r: redis.Redis) -> int:
    by_machine = defaultdict(list)
    for entry_id, fields in entries:
        by_machine[fields[b"m"].decode()].append((entry_id, fields))

    now_ms = int(time.time() * 1000)
    score_rows, alert_rows = [], []
    for machine, jobs in by_machine.items():
        windows = np.stack(
            [np.frombuffer(f[b"w"], dtype="<f4").reshape(WINDOW, N_FEATURES) for _, f in jobs]
        )
        scores, flags, errors = scorer.score(machine, windows)
        for (_, f), score, flag, error in zip(jobs, scores, flags, errors):
            ts = int(f[b"t"])
            score_rows.append((machine, ts, float(score), bool(flag), now_ms - int(f[b"i"]), NAME))
            if flag:
                top = [int(i) for i in np.argsort(error)[-3:][::-1]]
                alert_rows.append((machine, ts, float(score), top))

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO scores (machine_id, ts, score, is_anomaly, latency_ms, worker) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            score_rows,
        )
        if alert_rows:
            cur.executemany(
                "INSERT INTO alerts (machine_id, ts, score, top_metrics) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                alert_rows,
            )
    conn.commit()

    # Only now is it safe to tell Redis the jobs are done.
    ids = [entry_id for entry_id, _ in entries]
    pipe = r.pipeline(transaction=False)
    pipe.xack(JOBS_STREAM, GROUP, *ids)
    pipe.xdel(JOBS_STREAM, *ids)
    pipe.execute()
    return len(entries)


def claim_abandoned(r: redis.Redis):
    """Take over jobs that another worker read but never acknowledged."""
    pending = r.xpending_range(JOBS_STREAM, GROUP, "-", "+", BATCH)
    stale = [
        p["message_id"]
        for p in pending
        if p["consumer"].decode() != NAME and p["time_since_delivered"] >= CLAIM_IDLE_MS
    ]
    if not stale:
        return []
    claimed = [e for e in r.xclaim(JOBS_STREAM, GROUP, NAME, CLAIM_IDLE_MS, stale) if e[0] and e[1]]
    # A pending ID whose entry no longer exists has nothing left to process.
    gone = set(stale) - {entry_id for entry_id, _ in claimed}
    if gone:
        r.xack(JOBS_STREAM, GROUP, *gone)
    return claimed


def main():
    r = redis.from_url(REDIS_URL)
    try:
        r.xgroup_create(JOBS_STREAM, GROUP, id="0", mkstream=True)
    except redis.ResponseError as err:
        if "BUSYGROUP" not in str(err):
            raise

    scorer = Scorer()
    conn = db.connect()
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    print(f"{NAME} started", flush=True)
    last_claim = 0.0
    while running:
        if time.time() - last_claim > 2:
            last_claim = time.time()
            abandoned = claim_abandoned(r)
            if abandoned:
                print(f"{NAME} took over {len(abandoned)} abandoned jobs", flush=True)
                process(abandoned, scorer, conn, r)

        response = r.xreadgroup(GROUP, NAME, {JOBS_STREAM: ">"}, count=BATCH, block=500)
        if response:
            process(response[0][1], scorer, conn, r)


if __name__ == "__main__":
    main()
