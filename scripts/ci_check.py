"""Used by CI after a replay: waits until the pipeline has scored every
reading it should have, and fails if it never gets there.

Reads only the public API, so it works against the Docker stack as well.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

WINDOW = 50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--machines", type=int, required=True)
    parser.add_argument("--readings", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    expected = args.machines * (args.readings - WINDOW + 1)
    stats = {}
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        try:
            stats = httpx.get(f"{args.url}/stats", timeout=10).json()
            if stats["readings_scored"] >= expected:
                break
        except httpx.HTTPError:
            pass
        time.sleep(2)

    scored = stats.get("readings_scored", 0)
    print(f"expected {expected} scores, pipeline has {scored}")

    replay = Path("replay.json")
    if replay.exists():
        replicas = json.loads(replay.read_text())["replicas"]
        print(f"requests per API replica: {replicas}")
        if len(replicas) < 2:
            sys.exit("load balancer sent every request to one replica")

    if scored != expected:
        sys.exit(f"{expected - scored} readings were not scored")
    print("OK: nothing lost")


if __name__ == "__main__":
    main()
