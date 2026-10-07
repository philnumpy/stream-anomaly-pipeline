"""Ingestion API. Several identical copies run behind the Nginx load balancer.

A replica keeps nothing in its own memory between requests: sliding windows,
rate-limit counters and the job queue all live in Redis. That is what makes
it safe to run any number of replicas and to lose one mid-flight.
"""

import os
import time
from contextlib import asynccontextmanager

import numpy as np
import redis.asyncio as redis
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import db
from .config import (
    JOBS_STREAM,
    MAX_BACKLOG,
    N_FEATURES,
    RATE_LIMIT_PER_SECOND,
    REDIS_URL,
    ROOT,
    WINDOW,
)

REPLICA = os.environ.get("REPLICA_NAME", "api")

# Runs atomically inside Redis. Appends one reading to its machine's sliding
# window and, once the window is full, queues a scoring job that carries the
# whole window, so any worker can score it without other state.
#
# A reading whose timestamp is not newer than the last one seen for that
# machine is ignored. This makes ingestion idempotent: a client (or Nginx)
# can safely retry a request whose response was lost.
PUSH_READING = """
local last = tonumber(redis.call('GET', KEYS[2]) or '-1')
local ts = tonumber(ARGV[2])
if ts <= last then return -1 end
redis.call('SET', KEYS[2], ts)
redis.call('RPUSH', KEYS[1], ARGV[4])
redis.call('LTRIM', KEYS[1], -tonumber(ARGV[5]), -1)
if redis.call('LLEN', KEYS[1]) < tonumber(ARGV[5]) then return 0 end
local window = redis.call('LRANGE', KEYS[1], 0, -1)
redis.call('XADD', KEYS[3], '*', 'm', ARGV[1], 't', ARGV[2], 'i', ARGV[3], 'w', table.concat(window))
return 1
"""


class Reading(BaseModel):
    ts: int
    values: list[float]


class ReadingBatch(BaseModel):
    machine_id: str
    readings: list[Reading]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = redis.from_url(REDIS_URL)
    app.state.push = app.state.redis.register_script(PUSH_READING)
    yield
    await app.state.redis.aclose()


app = FastAPI(lifespan=lifespan)


@app.post("/readings")
async def post_readings(batch: ReadingBatch, x_client_id: str = Header(default="anonymous")):
    r: redis.Redis = app.state.redis
    now_ms = int(time.time() * 1000)

    rate_key = f"rate:{x_client_id}:{now_ms // 1000}"
    async with r.pipeline(transaction=False) as pipe:
        pipe.incr(rate_key)
        pipe.expire(rate_key, 2)
        pipe.xlen(JOBS_STREAM)
        requests_this_second, _, backlog = await pipe.execute()

    if requests_this_second > RATE_LIMIT_PER_SECOND:
        raise HTTPException(429, "rate limit exceeded")
    if backlog > MAX_BACKLOG:
        raise HTTPException(503, "scoring backlog is full, retry later")

    machine = batch.machine_id
    keys = [f"window:{machine}", f"last_ts:{machine}", JOBS_STREAM]
    async with r.pipeline(transaction=False) as pipe:
        for reading in batch.readings:
            if len(reading.values) != N_FEATURES:
                raise HTTPException(422, f"expected {N_FEATURES} values per reading")
            payload = np.asarray(reading.values, dtype="<f4").tobytes()
            await app.state.push(
                keys=keys, args=[machine, reading.ts, now_ms, payload, WINDOW], client=pipe
            )
        outcomes = await pipe.execute()

    return {
        "replica": REPLICA,
        "accepted": sum(1 for o in outcomes if o >= 0),
        "duplicates": sum(1 for o in outcomes if o == -1),
        "jobs_queued": sum(1 for o in outcomes if o == 1),
    }


@app.get("/health")
async def health():
    await app.state.redis.ping()
    return {"status": "ok", "replica": REPLICA}


# The read endpoints below are plain `def`, so FastAPI runs them in its thread
# pool and the blocking database driver does not stall the event loop.


@app.get("/alerts")
def alerts(machine_id: str | None = None, limit: int = 50):
    query = "SELECT machine_id, ts, score, top_metrics, created_at FROM alerts"
    params: list = []
    if machine_id:
        query += " WHERE machine_id = %s"
        params.append(machine_id)
    query += " ORDER BY created_at DESC, ts DESC LIMIT %s"
    params.append(min(limit, 500))
    with db.connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [
        {"machine_id": m, "ts": ts, "score": score, "top_metrics": top, "created_at": created}
        for m, ts, score, top, created in rows
    ]


@app.get("/stats")
def stats():
    with db.connect() as conn:
        scored, anomalies = conn.execute(
            "SELECT count(*), count(*) FILTER (WHERE is_anomaly) FROM scores"
        ).fetchone()
        per_machine = conn.execute(
            "SELECT machine_id, count(*) FROM alerts GROUP BY machine_id ORDER BY count(*) DESC LIMIT 10"
        ).fetchall()
    return {
        "replica": REPLICA,
        "readings_scored": scored,
        "anomalies": anomalies,
        "top_machines_by_alerts": [{"machine_id": m, "alerts": n} for m, n in per_machine],
    }


@app.get("/")
def dashboard():
    return FileResponse(ROOT / "app" / "dashboard.html")
