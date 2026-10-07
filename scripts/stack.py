"""Start and stop the whole stack as local processes (no Docker needed).

    python scripts/stack.py up [--apis 3] [--workers 3]
    python scripts/stack.py down
    python scripts/stack.py status
    python scripts/stack.py kill worker-2      # hard kill, for failure tests
    python scripts/stack.py start worker-2

Expects Redis and Nginx unpacked under tools/ and a PostgreSQL data folder at
tools/pgdata (see README, "Running without Docker").
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TOOLS = ROOT / "tools"
LOGS = ROOT / "logs"
STATE = TOOLS / "stack.json"
PG_BIN = Path(os.environ.get("PG_BIN", r"C:\Program Files\PostgreSQL\17\bin"))
NGINX_DIR = next(TOOLS.glob("nginx-*"), None)

REDIS_PORT, PG_PORT, LB_PORT, API_BASE_PORT = 6380, 5433, 8080, 8001


def command(name: str) -> list[str]:
    kind, _, index = name.partition("-")
    if kind == "redis":
        return [str(TOOLS / "redis" / "redis-server.exe"), "--port", str(REDIS_PORT),
                "--save", "", "--appendonly", "no", "--dir", str(TOOLS)]
    if kind == "postgres":
        return [str(PG_BIN / "postgres.exe"), "-D", str(TOOLS / "pgdata"), "-p", str(PG_PORT)]
    if kind == "nginx":
        return [str(NGINX_DIR / "nginx.exe"), "-p", str(NGINX_DIR),
                "-c", str(ROOT / "nginx" / "nginx.native.conf"), "-g", "daemon off;"]
    if kind == "api":
        return [sys.executable, "-m", "uvicorn", "app.ingest_api:app", "--host", "127.0.0.1",
                "--port", str(API_BASE_PORT + int(index) - 1), "--log-level", "warning"]
    if kind == "worker":
        return [sys.executable, "-m", "app.worker"]
    raise SystemExit(f"unknown process {name}")


def load() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2))


def alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
    return str(pid) in out.stdout


def start(name: str) -> int:
    LOGS.mkdir(exist_ok=True)
    env = {**os.environ, "REPLICA_NAME": name, "WORKER_NAME": name, "PYTHONUNBUFFERED": "1"}
    log = open(LOGS / f"{name}.log", "ab")
    proc = subprocess.Popen(command(name), cwd=ROOT, env=env, stdout=log, stderr=log,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    state = load()
    state[name] = proc.pid
    save(state)
    return proc.pid


def kill(name: str) -> None:
    state = load()
    pid = state.pop(name, None)
    if pid:
        # /T also takes child processes (Nginx and PostgreSQL fork workers).
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
    save(state)


def wait_for(check, what: str, seconds: int = 30) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            check()
            return
        except Exception:
            time.sleep(0.5)
    raise SystemExit(f"{what} did not come up within {seconds}s; see logs/")


def up(apis: int, workers: int) -> None:
    import psycopg
    import redis

    from app import db

    start("redis")
    start("postgres")
    wait_for(lambda: redis.Redis(port=REDIS_PORT, protocol=2).ping(), "Redis")

    def ensure_database():
        with psycopg.connect(f"postgresql://anomaly@127.0.0.1:{PG_PORT}/postgres", autocommit=True) as conn:
            if not conn.execute("SELECT 1 FROM pg_database WHERE datname = 'anomaly'").fetchone():
                conn.execute("CREATE DATABASE anomaly")

    wait_for(ensure_database, "PostgreSQL")
    thresholds = json.loads((ROOT / "artifacts" / "thresholds.json").read_text())
    db.init_schema(thresholds)

    for i in range(1, apis + 1):
        start(f"api-{i}")
    for i in range(1, workers + 1):
        start(f"worker-{i}")
    start("nginx")

    import httpx

    wait_for(lambda: httpx.get(f"http://127.0.0.1:{LB_PORT}/health").raise_for_status(), "API behind Nginx", 60)
    print(f"up: {apis} API replicas, {workers} workers, load balancer on http://127.0.0.1:{LB_PORT}")


def down() -> None:
    for name in list(load()):
        kill(name)
    print("down")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["up", "down", "status", "kill", "start"])
    parser.add_argument("name", nargs="?")
    parser.add_argument("--apis", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    if args.action == "up":
        down()
        up(args.apis, args.workers)
    elif args.action == "down":
        down()
    elif args.action == "status":
        for name, pid in load().items():
            print(f"{name:12} pid {pid:<7} {'running' if alive(pid) else 'DEAD'}")
    elif args.action == "kill":
        kill(args.name)
        print(f"killed {args.name}")
    elif args.action == "start":
        print(f"started {args.name} (pid {start(args.name)})")


if __name__ == "__main__":
    main()
