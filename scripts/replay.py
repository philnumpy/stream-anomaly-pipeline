"""Load generator: replays SMD test files as live machines sending readings.

Each machine is one concurrent client that posts its readings in order, the
way a real monitoring agent would. Requests that fail (connection refused,
429, 503) are retried, which is safe because ingestion is idempotent.

    python scripts/replay.py --machines 8 --readings 2000 --batch 10
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "data_src" / "ServerMachineDataset" / "test"


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0


async def run_machine(client, url, machine, rows, batch, interval, stats):
    for start in range(0, len(rows), batch):
        body = {
            "machine_id": machine,
            "readings": [
                {"ts": start + i, "values": row} for i, row in enumerate(rows[start : start + batch])
            ],
        }
        while True:
            began = time.perf_counter()
            try:
                response = await client.post(url, json=body, headers={"X-Client-Id": machine})
            except httpx.HTTPError:
                stats["retries"] += 1
                await asyncio.sleep(0.2)
                continue
            if response.status_code == 200:
                stats["latencies_ms"].append((time.perf_counter() - began) * 1000)
                stats["replicas"][response.json()["replica"]] = (
                    stats["replicas"].get(response.json()["replica"], 0) + 1
                )
                break
            stats["retries"] += 1
            stats["status"][response.status_code] = stats["status"].get(response.status_code, 0) + 1
            await asyncio.sleep(0.2)
        if interval:
            await asyncio.sleep(interval)


async def replay(url: str, machines: list[str], readings: int, batch: int, interval: float) -> dict:
    data = {
        m: np.loadtxt(TEST / f"{m}.txt", delimiter=",", dtype=np.float32, max_rows=readings).tolist()
        for m in machines
    }
    stats = {"latencies_ms": [], "retries": 0, "status": {}, "replicas": {}}
    limits = httpx.Limits(max_connections=len(machines) * 2)
    began = time.perf_counter()
    async with httpx.AsyncClient(limits=limits, timeout=10) as client:
        await asyncio.gather(
            *(run_machine(client, url, m, rows, batch, interval, stats) for m, rows in data.items())
        )
    seconds = time.perf_counter() - began
    sent = sum(len(rows) for rows in data.values())
    latencies = stats.pop("latencies_ms")
    return {
        "machines": len(machines),
        "readings_sent": sent,
        "seconds": round(seconds, 2),
        "readings_per_second": round(sent / seconds, 1),
        "requests_per_second": round(len(latencies) / seconds, 1),
        "request_latency_ms": {
            "p50": round(percentile(latencies, 50), 1),
            "p95": round(percentile(latencies, 95), 1),
            "p99": round(percentile(latencies, 99), 1),
        },
        **stats,
    }


def pick_machines(count: int) -> list[str]:
    trained = json.loads((ROOT / "artifacts" / "thresholds.json").read_text())
    return sorted(trained)[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080/readings")
    parser.add_argument("--machines", type=int, default=8)
    parser.add_argument("--readings", type=int, default=2000, help="readings per machine")
    parser.add_argument("--batch", type=int, default=10, help="readings per request")
    parser.add_argument("--interval", type=float, default=0.0, help="pause between requests (s)")
    args = parser.parse_args()
    result = asyncio.run(
        replay(args.url, pick_machines(args.machines), args.readings, args.batch, args.interval)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
