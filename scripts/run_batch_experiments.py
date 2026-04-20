import argparse
import csv
import datetime as dt
import re
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "ServerMachineDataset" / "train"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = RESULTS_DIR / "logs"


MODEL_COMMANDS = {
    "iforest": [
        "python",
        "train_iforest.py",
        "--window",
        "10",
        "--stride",
        "1",
        "--smooth-k",
        "5",
        "--percentiles",
        "99.0,99.5,99.9",
    ],
    "lstm": [
        "python",
        "train_lstm.py",
        "--window",
        "10",
        "--stride",
        "1",
        "--val-ratio",
        "0.1",
        "--hidden-size",
        "64",
        "--epochs",
        "20",
        "--patience",
        "5",
        "--batch-size",
        "64",
        "--smooth-k",
        "5",
        "--percentiles",
        "99.0,99.5,99.9",
        "--device",
        "cpu",
    ],
    "gcn": [
        "python",
        "train_gcn.py",
        "--window",
        "10",
        "--stride",
        "1",
        "--val-ratio",
        "0.1",
        "--corr-threshold",
        "0.8",
        "--corr-top-k",
        "6",
        "--hidden-dim",
        "32",
        "--embed-dim",
        "8",
        "--epochs",
        "20",
        "--patience",
        "5",
        "--batch-size",
        "64",
        "--smooth-k",
        "5",
        "--percentiles",
        "99.0,99.5,99.9",
        "--device",
        "cpu",
    ],
    "hier_hgnn": [
        "python",
        "train_hier_hgnn.py",
        "--window",
        "10",
        "--stride",
        "1",
        # KEY CHANGE 1: use global graph mode by default for robust batch runs.
        "--graph-mode",
        "global",
        "--hidden-dim",
        "24",
        "--embed-dim",
        "6",
        "--epochs",
        "24",
        "--patience",
        "4",
        "--lr",
        "7e-4",
        "--dropout",
        "0.30",
        "--weight-decay",
        "1e-3",
        "--latent-noise-std",
        "0.08",
        "--latent-reg-weight",
        "2e-3",
        "--denoise-std",
        "0.05",
        "--global-threshold",
        "0.75",
        "--global-top-k",
        "8",
        "--temporal-loss-weight",
        "0.1",
        # KEY CHANGE 2: explicitly set score weights for consistent parsing/eval behavior.
        "--score-weights",
        "0.85,0.1,0.05",
        "--smooth-k",
        "3",
        "--percentiles",
        "96.0,97.0,98.0,99.0,99.5",
        "--device",
        "cpu",
    ],
}


def parse_metric(output: str, label: str) -> float | None:
    pattern = re.compile(rf"{re.escape(label)}\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
    m = pattern.search(output)
    return float(m.group(1)) if m else None


def parse_int_metric(output: str, label: str) -> int | None:
    pattern = re.compile(rf"{re.escape(label)}\s*:\s*(\d+)")
    m = pattern.search(output)
    return int(m.group(1)) if m else None


# KEY CHANGE 3: multi-label parsers for robust compatibility across script output styles.
def parse_metric_multi(output: str, labels: list[str]) -> float | None:
    for label in labels:
        value = parse_metric(output, label)
        if value is not None:
            return value
    return None


def parse_int_metric_multi(output: str, labels: list[str]) -> int | None:
    for label in labels:
        value = parse_int_metric(output, label)
        if value is not None:
            return value
    return None


def parse_run_output(output: str) -> dict:
    # KEY CHANGE 4: parse both "Best ..." and plain metric labels.
    return {
        "best_percentile": parse_metric_multi(output, ["Best percentile", "Percentile"]),
        "best_threshold": parse_metric_multi(output, ["Best threshold", "Threshold"]),
        "precision": parse_metric_multi(output, ["Best precision", "Precision"]),
        "recall": parse_metric_multi(output, ["Best recall", "Recall"]),
        "f1": parse_metric_multi(output, ["Best F1", "F1-score"]),
        "pred_anomalies": parse_int_metric_multi(output, ["Pred anomalies", "Pred count"]),
        "true_anomalies": parse_int_metric_multi(output, ["True anomalies", "True count"]),
    }


def discover_all_machines() -> list[str]:
    if not TRAIN_DIR.exists():
        return []
    machines = []
    for p in sorted(TRAIN_DIR.glob("machine-*.txt")):
        machines.append(p.stem)
    return machines


def build_tasks(models: list[str], machines: list[str]) -> list[tuple[str, str]]:
    tasks = []
    for model in models:
        for machine in machines:
            tasks.append((model, machine))
    return tasks


def run_one(model: str, machine: str, timeout_sec: int) -> dict:
    cmd = MODEL_COMMANDS[model] + ["--machine", machine]
    started_at = dt.datetime.now().isoformat(timespec="seconds")
    start_t = time.perf_counter()

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_sec,
    )
    duration_sec = time.perf_counter() - start_t
    full_output = (proc.stdout or "") + "\n" + (proc.stderr or "")

    log_name = f"{model}__{machine}.log"
    log_path = LOG_DIR / log_name
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# command: {' '.join(cmd)}\n")
        f.write(f"# started_at: {started_at}\n")
        f.write(f"# returncode: {proc.returncode}\n")
        f.write(f"# duration_sec: {duration_sec:.3f}\n\n")
        f.write(full_output)

    parsed = parse_run_output(full_output)
    ok = proc.returncode == 0 and parsed["f1"] is not None

    return {
        "timestamp": started_at,
        "model": model,
        "machine": machine,
        "status": "ok" if ok else "failed",
        "returncode": proc.returncode,
        "duration_sec": round(duration_sec, 3),
        "command": " ".join(cmd),
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
        "error": "" if ok else f"run_failed_or_parse_failed(returncode={proc.returncode})",
        **parsed,
    }


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    headers = [
        "timestamp",
        "model",
        "machine",
        "status",
        "returncode",
        "duration_sec",
        "best_percentile",
        "best_threshold",
        "precision",
        "recall",
        "f1",
        "pred_anomalies",
        "true_anomalies",
        "log_path",
        "command",
        "error",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="Batch run anomaly detection experiments.")
    parser.add_argument(
        "--models",
        type=str,
        default="iforest,lstm,gcn,hier_hgnn",
        help="Comma-separated model keys: iforest,lstm,gcn,hier_hgnn",
    )
    parser.add_argument(
        "--machines",
        type=str,
        default="machine-1-1,machine-2-1,machine-3-1",
        help="Comma-separated machines. Ignored when --all-machines is set.",
    )
    parser.add_argument("--all-machines", action="store_true", help="Auto scan all SMD machines.")
    parser.add_argument("--timeout-sec", type=int, default=7200, help="Timeout per run.")
    parser.add_argument(
        "--results-csv",
        type=str,
        default="results/results.csv",
        help="Path (relative to project root) to save results CSV.",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in MODEL_COMMANDS:
            raise ValueError(f"Unknown model key: {m}. Available: {list(MODEL_COMMANDS.keys())}")

    if args.all_machines:
        machines = discover_all_machines()
    else:
        machines = [x.strip() for x in args.machines.split(",") if x.strip()]

    if not machines:
        raise ValueError("No machines found/provided.")

    tasks = build_tasks(models, machines)
    print(f"Total tasks: {len(tasks)}")
    for i, (model, machine) in enumerate(tasks, start=1):
        print(f"[{i}/{len(tasks)}] {model} @ {machine}")

    rows = []
    for i, (model, machine) in enumerate(tasks, start=1):
        print(f"\nRunning [{i}/{len(tasks)}]: model={model}, machine={machine}")
        try:
            row = run_one(model=model, machine=machine, timeout_sec=args.timeout_sec)
        except subprocess.TimeoutExpired as e:
            log_name = f"{model}__{machine}.log"
            log_path = LOG_DIR / log_name
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"TIMEOUT: {e}\n")
                if e.stdout:
                    f.write("\n--- stdout ---\n")
                    f.write(e.stdout)
                if e.stderr:
                    f.write("\n--- stderr ---\n")
                    f.write(e.stderr)
            row = {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "model": model,
                "machine": machine,
                "status": "failed",
                "returncode": -999,
                "duration_sec": None,
                "best_percentile": None,
                "best_threshold": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "pred_anomalies": None,
                "true_anomalies": None,
                "log_path": str(log_path.relative_to(PROJECT_ROOT)),
                "command": "TIMEOUT",
                "error": "timeout",
            }
        except Exception as e:
            row = {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "model": model,
                "machine": machine,
                "status": "failed",
                "returncode": -998,
                "duration_sec": None,
                "best_percentile": None,
                "best_threshold": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "pred_anomalies": None,
                "true_anomalies": None,
                "log_path": "",
                "command": "",
                "error": f"exception: {e}",
            }

        rows.append(row)
        print(
            f"status={row['status']}, "
            f"f1={row.get('f1')}, "
            f"log={row.get('log_path')}"
        )

    out_csv = PROJECT_ROOT / args.results_csv
    write_csv(rows, out_csv)
    print(f"\nSaved results to: {out_csv}")
    print(f"Logs dir: {LOG_DIR}")


if __name__ == "__main__":
    sys.exit(main())
