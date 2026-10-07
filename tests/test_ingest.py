"""Integration tests for the ingestion API. They need Redis and are skipped
when it is not reachable (start it with `python scripts/stack.py up`)."""

import uuid

import pytest
import redis
from fastapi.testclient import TestClient

from app.config import N_FEATURES, REDIS_URL, WINDOW
from app.ingest_api import app


@pytest.fixture(scope="module")
def client():
    try:
        redis.from_url(REDIS_URL).ping()
    except redis.RedisError:
        pytest.skip("Redis is not running")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def machine():
    name = f"test-{uuid.uuid4().hex[:8]}"
    yield name
    redis.from_url(REDIS_URL).delete(f"window:{name}", f"last_ts:{name}")


def readings(start: int, count: int) -> list[dict]:
    return [{"ts": ts, "values": [0.1] * N_FEATURES} for ts in range(start, start + count)]


def post(client, machine, batch):
    return client.post("/readings", json={"machine_id": machine, "readings": batch})


def test_no_job_until_the_window_is_full(client, machine):
    body = post(client, machine, readings(0, WINDOW - 1)).json()
    assert body["accepted"] == WINDOW - 1
    assert body["jobs_queued"] == 0


def test_every_reading_after_the_window_fills_queues_one_job(client, machine):
    post(client, machine, readings(0, WINDOW - 1))
    body = post(client, machine, readings(WINDOW - 1, 5)).json()
    assert body["jobs_queued"] == 5


def test_resending_the_same_readings_is_ignored(client, machine):
    batch = readings(0, WINDOW + 3)
    first = post(client, machine, batch).json()
    second = post(client, machine, batch).json()
    assert first["jobs_queued"] == 4
    assert second["accepted"] == 0
    assert second["duplicates"] == WINDOW + 3
    assert second["jobs_queued"] == 0


def test_reading_with_wrong_number_of_metrics_is_rejected(client, machine):
    response = post(client, machine, [{"ts": 0, "values": [0.1, 0.2]}])
    assert response.status_code == 422
