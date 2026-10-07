# Real-Time Stream Processing Pipeline with Deep Learning Anomaly Detection

Simulated servers send their monitoring metrics to a load-balanced ingestion
API. The readings flow through a Redis stream to a pool of workers, which
score every reading with an LSTM autoencoder and store anomalies in
PostgreSQL.

```
 machines (replayed SMD test data)
        |
        v
  Nginx load balancer  (round robin, failed replica skipped for 5 s)
        |
   +----+----+----+
   v         v    v
 api-1    api-2  api-3      FastAPI, stateless
   |         |    |
   +----+----+----+
        v
      Redis    sliding window per machine  |  rate-limit counters
        |      stream of scoring jobs
        v
 worker-1  worker-2  worker-3     one consumer group, ONNX Runtime
        |
        v
   PostgreSQL   machines, scores, alerts
```

## How a reading moves through the system

1. A machine posts a batch of readings (38 metrics each) to Nginx, which
   forwards it to one of the API replicas.
2. The replica checks the rate limit and the size of the job backlog, then
   runs one small Lua script in Redis per reading. The script appends the
   reading to that machine's sliding window and, once 50 readings are present,
   adds a scoring job containing the whole window to the `jobs` stream.
3. Workers read jobs from the stream as members of one consumer group, so each
   job goes to exactly one worker. A worker groups its jobs by machine, runs
   that machine's LSTM on the batch, and writes scores and alerts to
   PostgreSQL.
4. Only after the database commit does the worker acknowledge the jobs.

## Design decisions

**The API replicas keep no state.** Windows, rate-limit counters and the queue
are all in Redis. An in-memory rate limiter would allow three times the limit
with three replicas, and an in-memory window would be split across them. With
the state outside, any replica can serve any request and losing one loses
nothing.

**Each job carries its full window.** This costs memory (a job is about 7.6 KB
instead of 152 bytes), but it means a worker needs nothing except the job, so
jobs can be handed to any worker in any order. The alternative is what Kafka
does: partition the stream by machine so one worker sees a machine's readings
in order and keeps the window itself. That is more efficient and more
complicated; here simplicity won.

**Delivery is at-least-once, and writes are idempotent.** A worker that dies
after reading jobs but before acknowledging them leaves them in the group's
pending list. Every 2 seconds each worker looks for jobs pending longer than
10 seconds and takes them over. A job can therefore be scored twice, which is
harmless because `scores` and `alerts` are keyed on `(machine_id, ts)` and
inserts use `ON CONFLICT DO NOTHING`.

**Ingestion is idempotent too.** The Lua script ignores a reading whose
timestamp is not newer than the last one stored for that machine. So when a
replica dies mid-request, Nginx can resend the same POST to another replica
without the reading being counted twice.

**Backpressure instead of unbounded queues.** When more than 20,000 jobs are
waiting, the API answers 503 and the client retries later. The queue never
silently drops data and Redis memory stays bounded.

**Redis for what is temporary, PostgreSQL for what must last.** Windows and
the queue change constantly and can be rebuilt, so they live in memory.
Scores and alerts are the record of what happened, so they go to disk. Raw
readings are not stored at all, only one score per reading.

## The model

An LSTM autoencoder (`anomaly/model.py`): the encoder compresses a window of
50 readings into a 64-number vector and the decoder rebuilds the window from
it. It is trained only on each machine's normal period, so it rebuilds normal
behaviour well and anything else badly. The anomaly score of a reading is the
mean squared error between it and its reconstruction.

One model is trained per machine, exported to ONNX, and served with ONNX
Runtime, so the serving side does not need PyTorch.

The alert threshold is chosen without labels: the 99th percentile of scores
on the last 15% of the training period, which the model did not train on.

## Results

### Pipeline

Measured on one laptop (Intel i7-7600U, 2 cores / 4 threads, 16 GB RAM,
Windows 10) with every component and the load generator on the same machine,
running as local processes. Load: all 28 machines sending at once, 2,000
readings each (56,000 readings, 10 per request). Raw outputs are in
`results/bench_*.json`.

| Test | Result |
|---|---|
| Throughput, 3 APIs + 3 workers | 1,227 readings/s ingested, 1,170 scored/s |
| Detection delay (reading received to score ready) | p50 6 ms, p95 10 ms, p99 13 ms |
| Load balancing | 1,866 / 1,867 / 1,867 requests per replica |
| API replica killed mid-run, restarted after 5 s | 0 of 54,628 scores lost, 0 failed client requests |
| Worker killed mid-run, restarted after 5 s | 0 of 54,628 scores lost |
| 500 jobs abandoned by a fake crashed worker | all taken over by other workers, 0 lost |
| Live scores vs offline evaluation (11,804 readings) | identical to within 2e-8 |

**Replicas did not add throughput on this machine.** One API and one worker
ingested 1,470 readings/s, more than three of each (1,227/s). With two CPU
cores shared by Nginx, Redis, PostgreSQL, the APIs, the workers and the load
generator, extra processes only compete for the same cores. What the replicas
did buy here is fault tolerance and a tighter delay tail (p99 13 ms against
85 ms with a single worker). Showing a throughput gain needs the replicas on
separate cores or machines.

Request latency under this load was p50 121 ms and p95 682 ms, which reflects
28 clients saturating two cores, not the cost of one request: with 4 machines
sending, it was p50 20 ms and p95 35 ms.

### Model

All 28 machines, one model each (`results/smd_results.json`):

| Metric | Mean over 28 machines |
|---|---|
| Point-adjusted F1, best threshold | 0.92 |
| Plain F1, best threshold | 0.42 |
| Plain F1, threshold chosen without labels | 0.23 |
| AUROC | 0.79 |

How to read these:

- *Point-adjusted, best threshold* is the protocol most SMD papers report
  (OmniAnomaly reports 0.89). If any point of a true anomaly segment is
  flagged, the whole segment counts as detected, and the threshold is the one
  that maximises F1 on the test labels. Both choices flatter the model.
- *Plain F1 without labels* is what the running pipeline actually achieves:
  every point is judged on its own and the threshold comes from normal data
  only. It is much lower, and it varies from 0.03 to 0.80 across machines.
- The main weakness is the threshold, not the ranking. On a few machines the
  test period drifts away from the training period, so a threshold set on
  training data flags nearly everything (recall near 1, precision under 0.05).
  A threshold that adapts over time would be the next thing to try.

## Dataset

[Server Machine Dataset](https://github.com/NetManAIOps/OmniAnomaly) (SMD):
28 machines, 38 metrics, one reading per minute for about five weeks. The
first half is unlabelled training data; the second half is test data with
anomaly labels.

```
git clone --depth 1 --filter=blob:none --sparse https://github.com/NetManAIOps/OmniAnomaly.git data_src
cd data_src && git sparse-checkout set ServerMachineDataset
```

## How to run

Needs Python 3.11 or newer.

### 1. Get the code and the dataset

```
git clone https://github.com/philnumpy/stream-anomaly-pipeline.git
cd stream-anomaly-pipeline
pip install -r requirements.txt

git clone --depth 1 --filter=blob:none --sparse https://github.com/NetManAIOps/OmniAnomaly.git data_src
cd data_src
git sparse-checkout set ServerMachineDataset
cd ..
```

### 2. Train and evaluate the models (optional)

The trained ONNX models for all machines are already in `artifacts/`, so this
step is only needed to reproduce the accuracy numbers.

```
python -m anomaly.train_eval                          # all 28 machines
python -m anomaly.train_eval --machines machine-1-1   # or just one
```

Results are written to `results/smd_results.json`. A re-run skips machines
that are already done; add `--force` to retrain.

### 3. Start the pipeline

**Option A: Docker**

```
docker compose up --build --scale api=3 --scale worker=3
```

Note: the Compose setup mirrors Option B but has not been run end to end yet;
all results in this README come from Option B.

**Option B: local processes (Windows, no Docker)**

This is how the benchmarks below were run. One-time setup:

1. Install [PostgreSQL](https://www.postgresql.org/download/windows/). Only
   its programs are used; the pipeline runs its own separate database on port
   5433. If it is not installed in `C:\Program Files\PostgreSQL\17`, set the
   `PG_BIN` environment variable to its `bin` folder.
2. Download [Redis for Windows](https://github.com/tporadowski/redis/releases)
   (the zip) and unpack it to `tools\redis`.
3. Download [Nginx for Windows](https://nginx.org/en/download.html) and unpack
   it into `tools\`, so that `tools\nginx-<version>\nginx.exe` exists.
4. Create the database folder:

   ```
   "C:\Program Files\PostgreSQL\17\bin\initdb.exe" -D tools\pgdata -U anomaly -A trust -E UTF8
   ```

Then:

```
python scripts/stack.py up        # Redis, PostgreSQL, 3 APIs, 3 workers, Nginx
python scripts/stack.py status
```

### 4. Send data through it

With the stack running, open the dashboard at http://127.0.0.1:8080/ and run:

```
python scripts/replay.py --machines 8 --readings 2000
```

This replays eight machines' test data as live traffic; alerts appear on the
dashboard as they are detected.

### 5. Run the benchmarks

These use the local-process setup, because they kill and restart processes.

```
python scripts/bench.py verify --machines 2 --readings 3000    # live scores == offline scores
python scripts/bench.py load   --machines 8 --readings 2000    # throughput and latency
python scripts/bench.py fault  --kill worker-2                 # kill a worker mid-run
python scripts/bench.py fault  --kill api-2                    # kill an API replica mid-run
python scripts/bench.py load   --orphan 300                    # jobs abandoned by a crashed worker
```

Each one prints a summary and saves it under `results/`.

### 6. Stop

```
python scripts/stack.py down
```

## Layout

```
anomaly/model.py        LSTM autoencoder, windowing
anomaly/train_eval.py   training, evaluation, ONNX export
anomaly/metrics.py      F1, point-adjusted F1, AUROC
app/ingest_api.py       ingestion API (runs as N replicas)
app/worker.py           stream worker (runs as N consumers)
app/db.py               PostgreSQL schema
scripts/stack.py        start/stop/kill local processes
scripts/replay.py       load generator
scripts/bench.py        load, fault and correctness benchmarks
nginx/                  load balancer configuration
results/                benchmark outputs
```
