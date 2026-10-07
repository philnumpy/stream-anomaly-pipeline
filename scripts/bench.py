"""Benchmarks against a running stack (python scripts/stack.py up).

    python scripts/bench.py load   --name w3 --machines 8 --readings 2000
    python scripts/bench.py fault  --kill worker-2 --machines 8 --readings 2000
    python scripts/bench.py fault  --kill api-2    --machines 8 --readings 2000
    python scripts/bench.py verify --machines 2 --readings 3000

Every run starts from empty tables and an empty Redis, then checks that every
reading that should have been scored was scored exactly once.
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psycopg
import redis

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import DATABASE_URL, GROUP, JOBS_STREAM, REDIS_URL, WINDOW  # noqa: E402
from scripts.replay import pick_machines, replay  # noqa: E402

RESULTS = ROOT / "results"
URL = "http://127.0.0.1:8080/readings"


def reset() -> None:
    r = redis.from_url(REDIS_URL)
    keys = [k for pattern in ("window:*", "last_ts:*", "rate:*") for k in r.scan_iter(pattern)]
    if keys:
        r.delete(*keys)
    r.xtrim(JOBS_STREAM, maxlen=0)
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE scores, alerts")


def scored_count() -> int:
    with psycopg.connect(DATABASE_URL) as conn:
        return conn.execute("SELECT count(*) FROM scores").fetchone()[0]


def wait_until_scored(expected: int, timeout: int = 600) -> float:
    """Returns the time at which the last expected score landed."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if scored_count() >= expected:
            return time.perf_counter()
        time.sleep(0.25)
    return time.perf_counter()


def pipeline_summary(expected: int, began: float, finished: float) -> dict:
    with psycopg.connect(DATABASE_URL) as conn:
        scored, anomalies = conn.execute(
            "SELECT count(*), count(*) FILTER (WHERE is_anomaly) FROM scores"
        ).fetchone()
        p50, p95, p99 = conn.execute(
            "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms), "
            "percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms), "
            "percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms) FROM scores"
        ).fetchone()
        by_worker = dict(conn.execute("SELECT worker, count(*) FROM scores GROUP BY worker ORDER BY worker").fetchall())
    return {
        "expected_scores": expected,
        "scored": scored,
        "lost": expected - scored,
        "anomalies_flagged": anomalies,
        "end_to_end_seconds": round(finished - began, 2),
        "scored_per_second": round(scored / (finished - began), 1),
        "detection_delay_ms": {"p50": p50, "p95": p95, "p99": p99},
        "scored_by_worker": by_worker,
        "backlog_left": redis.from_url(REDIS_URL).xlen(JOBS_STREAM),
    }


async def run(args, kill: str | None = None) -> dict:
    machines = pick_machines(args.machines)
    expected = len(machines) * max(0, args.readings - WINDOW + 1)
    reset()

    async def killer():
        # Kill once roughly a third of the work is done, restart 5 s later.
        while scored_count() < expected / 3:
            await asyncio.sleep(0.25)
        subprocess.run([sys.executable, "scripts/stack.py", "kill", kill], cwd=ROOT, capture_output=True)
        killed_at = scored_count()
        await asyncio.sleep(5)
        subprocess.run([sys.executable, "scripts/stack.py", "start", kill], cwd=ROOT, capture_output=True)
        return killed_at

    async def orphaner():
        # Simulates a worker that dies holding jobs: a throwaway consumer
        # reads some jobs and never acknowledges them. The real workers have
        # to notice and take them over, or those readings are never scored.
        r = redis.from_url(REDIS_URL)
        while scored_count() < expected / 3:
            await asyncio.sleep(0.25)
        taken = 0
        while taken < args.orphan:
            got = await asyncio.to_thread(
                r.xreadgroup, GROUP, "crashed-worker", {JOBS_STREAM: ">"}, args.orphan - taken, 200
            )
            taken += len(got[0][1]) if got else 0
        return taken

    began = time.perf_counter()
    tasks = [replay(URL, machines, args.readings, args.batch, args.interval)]
    if kill:
        tasks.append(killer())
    elif args.orphan:
        tasks.append(orphaner())
    outcome = await asyncio.gather(*tasks)
    finished = await asyncio.to_thread(wait_until_scored, expected)

    result = {"ingest": outcome[0], "pipeline": pipeline_summary(expected, began, finished)}
    if kill:
        result["killed"] = {"process": kill, "after_scores": outcome[1], "restarted_after_s": 5}
    elif args.orphan:
        result["orphaned_jobs"] = outcome[1]
    return result


def verify(args) -> dict:
    """End-to-end correctness: scores produced by the live pipeline must equal
    the scores the offline evaluation computed for the same readings."""
    result = asyncio.run(run(args))
    machines = pick_machines(args.machines)
    worst = 0.0
    compared = 0
    with psycopg.connect(DATABASE_URL) as conn:
        for machine in machines:
            rows = conn.execute("SELECT ts, score FROM scores WHERE machine_id = %s ORDER BY ts", (machine,)).fetchall()
            offline = np.load(ROOT / "artifacts" / f"{machine}.scores.npy")
            ts = np.array([r[0] for r in rows])
            online = np.array([r[1] for r in rows], dtype=np.float32)
            worst = max(worst, float(np.max(np.abs(online - offline[ts]))))
            compared += len(rows)
    result["verification"] = {"scores_compared": compared, "max_abs_difference": worst}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["load", "fault", "verify"])
    parser.add_argument("--name", help="label for the results file")
    parser.add_argument("--kill", help="process to kill mid-run (fault mode)")
    parser.add_argument("--machines", type=int, default=8)
    parser.add_argument("--readings", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--interval", type=float, default=0.0)
    parser.add_argument("--orphan", type=int, default=0,
                        help="load mode: have a fake consumer take this many jobs and abandon them")
    args = parser.parse_args()

    if args.mode == "verify":
        result = verify(args)
    else:
        result = asyncio.run(run(args, kill=args.kill if args.mode == "fault" else None))

    name = args.name or (f"fault_{args.kill}" if args.mode == "fault" else args.mode)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"bench_{name}.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
