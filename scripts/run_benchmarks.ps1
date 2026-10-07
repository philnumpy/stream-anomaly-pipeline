# Runs the full benchmark suite and leaves the results in results/.
# Close other heavy programs first: everything, including the load
# generator, shares this machine's CPU.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

# Correctness: live scores must equal offline scores.
python scripts/stack.py up --apis 3 --workers 3
python scripts/bench.py verify --machines 4 --readings 3000 --name verify | Out-Null

# Throughput and latency, all 28 machines, 3 APIs + 3 workers.
python scripts/bench.py load --machines 28 --readings 2000 --name load_3api_3worker | Out-Null

# Failures under load.
python scripts/bench.py fault --kill worker-2 --machines 28 --readings 2000 | Out-Null
python scripts/bench.py fault --kill api-2 --machines 28 --readings 2000 | Out-Null
python scripts/bench.py load --orphan 500 --machines 28 --readings 2000 --name orphaned_jobs | Out-Null

# Scaling comparison: one API and one worker.
python scripts/stack.py up --apis 1 --workers 1
python scripts/bench.py load --machines 28 --readings 2000 --name load_1api_1worker | Out-Null

python scripts/stack.py down
Get-ChildItem results
