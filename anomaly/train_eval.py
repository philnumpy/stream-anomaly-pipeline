"""Train one LSTM autoencoder per SMD machine, evaluate it, export it to ONNX.

    python -m anomaly.train_eval                       all 28 machines
    python -m anomaly.train_eval --machines machine-1-1 machine-1-2

Results are written after every machine, so a stopped run keeps its progress
and a re-run skips machines that are already done (use --force to redo).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .metrics import evaluate
from .model import WINDOW, LastStepReconstruction, LSTMAutoencoder, make_windows, preprocess

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_src" / "ServerMachineDataset"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results" / "smd_results.json"

EPOCHS = 10
BATCH = 64
# Consecutive windows overlap in 49 of 50 readings, so training on every 4th
# one loses little and is 4x cheaper. Scoring always uses every window.
TRAIN_STRIDE = 4
LR = 1e-3
VAL_FRACTION = 0.15
# The alert threshold is set without labels: the 99th percentile of anomaly
# scores on held-out normal data (the last 15% of the training period).
THRESHOLD_QUANTILE = 0.99


def load(split: str, machine: str) -> np.ndarray:
    return np.loadtxt(DATA / split / f"{machine}.txt", delimiter=",", dtype=np.float32)


@torch.no_grad()
def last_step_scores(model: nn.Module, x: np.ndarray) -> np.ndarray:
    """Anomaly score for every reading that has a full window behind it:
    mean squared error between the reading and its reconstruction."""
    model.eval()
    windows = make_windows(x)
    out = []
    for i in range(0, len(windows), 1024):
        batch = torch.from_numpy(np.ascontiguousarray(windows[i : i + 1024]))
        recon = model(batch)[:, -1, :]
        out.append(((recon - batch[:, -1, :]) ** 2).mean(dim=1).numpy())
    return np.concatenate(out)


def train_machine(machine: str, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    started = time.time()

    train = preprocess(load("train", machine))
    test = preprocess(load("test", machine))
    labels = np.loadtxt(DATA / "test_label" / f"{machine}.txt", dtype=np.int8)

    split = int(len(train) * (1 - VAL_FRACTION))
    fit, val = train[:split], train[split:]
    fit_windows = make_windows(fit)[::TRAIN_STRIDE]

    model = LSTMAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val, best_state = float("inf"), None
    for epoch in range(EPOCHS):
        model.train()
        order = np.random.permutation(len(fit_windows))
        for i in range(0, len(order), BATCH):
            idx = np.sort(order[i : i + BATCH])
            batch = torch.from_numpy(np.ascontiguousarray(fit_windows[idx]))
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()
        val_loss = float(last_step_scores(model, val).mean())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    threshold = float(np.quantile(last_step_scores(model, val), THRESHOLD_QUANTILE))

    # The first WINDOW-1 test readings have no full window; score them 0.
    scores = np.concatenate([np.zeros(WINDOW - 1, dtype=np.float32), last_step_scores(model, test)])
    result = evaluate(scores, labels, threshold)
    result.update(machine=machine, val_loss=best_val, train_seconds=round(time.time() - started, 1))

    ARTIFACTS.mkdir(exist_ok=True)
    np.save(ARTIFACTS / f"{machine}.scores.npy", scores)
    torch.onnx.export(
        LastStepReconstruction(model).eval(),
        torch.zeros(1, WINDOW, train.shape[1]),
        str(ARTIFACTS / f"{machine}.onnx"),
        input_names=["window"],
        output_names=["reconstruction"],
        dynamic_axes={"window": {0: "batch"}, "reconstruction": {0: "batch"}},
        dynamo=False,
    )
    return result


def summarise(per_machine: list[dict]) -> dict:
    def mean(path: list[str]) -> float:
        values = []
        for r in per_machine:
            for key in path:
                r = r[key]
            values.append(r)
        return round(float(np.mean(values)), 4)

    # Pooled F1: add up the counts across machines, then compute one F1.
    tp = sum(r["plain"]["tp"] for r in per_machine)
    fp = sum(r["plain"]["fp"] for r in per_machine)
    fn = sum(r["plain"]["fn"] for r in per_machine)
    pooled_p = tp / (tp + fp) if tp + fp else 0.0
    pooled_r = tp / (tp + fn) if tp + fn else 0.0
    pooled_f1 = 2 * pooled_p * pooled_r / (pooled_p + pooled_r) if pooled_p + pooled_r else 0.0

    return {
        "machines": len(per_machine),
        "threshold_rule": f"{THRESHOLD_QUANTILE} quantile of held-out normal scores (no labels)",
        "label_free_threshold": {
            "f1_mean": mean(["plain", "f1"]),
            "precision_mean": mean(["plain", "precision"]),
            "recall_mean": mean(["plain", "recall"]),
            "f1_pooled": round(pooled_f1, 4),
            "point_adjusted_f1_mean": mean(["point_adjusted", "f1"]),
        },
        "oracle_threshold_upper_bound": {
            "f1_mean": mean(["best_plain", "f1"]),
            "point_adjusted_f1_mean": mean(["best_point_adjusted", "f1"]),
        },
        "auroc_mean": mean(["auroc"]),
        "auprc_mean": mean(["auprc"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--machines", nargs="*")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    machines = args.machines or sorted(p.stem for p in (DATA / "train").glob("*.txt"))
    RESULTS.parent.mkdir(exist_ok=True)
    done = {}
    if RESULTS.exists() and not args.force:
        done = {r["machine"]: r for r in json.loads(RESULTS.read_text())["per_machine"]}

    for machine in machines:
        if machine in done:
            print(f"{machine}: already done, skipping")
            continue
        result = train_machine(machine)
        done[machine] = result
        per_machine = [done[m] for m in sorted(done)]
        RESULTS.write_text(json.dumps({"summary": summarise(per_machine), "per_machine": per_machine}, indent=2))
        thresholds = {m: done[m]["threshold"] for m in sorted(done)}
        (ARTIFACTS / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
        print(
            f"{machine}: F1 {result['plain']['f1']:.3f} "
            f"(P {result['plain']['precision']:.3f}, R {result['plain']['recall']:.3f}), "
            f"best F1 {result['best_plain']['f1']:.3f}, "
            f"adjusted best F1 {result['best_point_adjusted']['f1']:.3f}, "
            f"AUROC {result['auroc']:.3f}, {result['train_seconds']}s",
            flush=True,
        )

    print(json.dumps(summarise([done[m] for m in sorted(done)]), indent=2))


if __name__ == "__main__":
    main()
