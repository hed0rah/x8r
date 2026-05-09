#!/usr/bin/env python3
"""
threshold_sweep.py

Sweep BPE_HEAP_THRESHOLD over {0, 16, 32, 64, 100, 200, 999999},
rebuild x8r at each threshold, verify correctness (golden + fuzz),
benchmark the six key corpus files, and produce CSV + chart.

Usage:
    source .venv/bin/activate
    python3 bench/threshold_sweep.py
"""
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
BUILD = ROOT / "build"
X8R_BIN = BUILD / "x8r"
BENCH_RUN = ROOT / "bench" / "run.py"

# Threshold values to sweep.
# 0 = always-heap (bookend), 999999 = always-linear (bookend).
THRESHOLDS = [0, 16, 32, 64, 100, 200, 999999]

# Target files for the sweep (6 specified in the task).
TARGET_FILES = [
    "cjk_dense.txt",
    "long_words.txt",
    "deep_indent.txt",
    "markdown_rules.txt",
    "utf8_prose_small.txt",
    "ascii_code_small.txt",
]

# Warm iterations for bench/run.py (1 discarded for prime, 9 recorded).
ITERS = 9

# Output paths
OUT_DIR = ROOT / "bench" / "profiles"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "threshold_sweep.csv"
PNG_PATH = OUT_DIR / "threshold_sweep.png"


def run(cmd, **kwargs):
    """Run a command, return stdout. Die on failure."""
    print(f"  $ {' '.join(str(c) for c in cmd)}", file=sys.stderr)
    r = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if r.returncode != 0:
        print(f"  FAILED (rc={r.returncode})", file=sys.stderr)
        if r.stderr:
            print(r.stderr[:2000], file=sys.stderr)
        sys.exit(1)
    return r.stdout


def rebuild_x8r(threshold: int):
    """Clean-rebuild x8r with the given BPE_HEAP_THRESHOLD."""
    print(f"\n=== Rebuilding x8r with BPE_HEAP_THRESHOLD={threshold} ===", file=sys.stderr)
    # Full clean
    if BUILD.exists():
        shutil.rmtree(BUILD)

    cflags = (
        "-std=c11 -O3 -march=x86-64-v3 "
        "-Wall -Wextra -Wpedantic -Wno-unused-parameter "
        f"-DBPE_HEAP_THRESHOLD={threshold}"
    )
    env = dict(os.environ, CFLAGS=cflags)
    run(["make", "-j", str(os.cpu_count() or 4), "-s"], env=env)
    assert X8R_BIN.exists(), "build did not produce x8r binary"


def verify_correctness():
    """Run golden tests and fuzz. Exit on failure."""
    print("  Verifying golden tests...", file=sys.stderr)
    out = subprocess.run(
        ["bash", str(ROOT / "tests" / "run_golden.sh")],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        print("  GOLDEN TESTS FAILED:", file=sys.stderr)
        print(out.stdout, file=sys.stderr)
        print(out.stderr, file=sys.stderr)
        return False

    # Parse golden output: must contain "5 passed, 0 failed"
    if "5 passed, 0 failed" not in out.stdout:
        print(f"  GOLDEN TESTS INCOMPLETE:\n{out.stdout}", file=sys.stderr)
        return False

    print("  Verifying fuzz (500 iterations)...", file=sys.stderr)
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "fuzz_vs_tiktoken.py"), "500"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        print("  FUZZ FAILED:", file=sys.stderr)
        print(out.stdout, file=sys.stderr)
        print(out.stderr, file=sys.stderr)
        return False

    if "no mismatches" not in out.stdout:
        print(f"  FUZZ HAS MISMATCHES:\n{out.stdout}", file=sys.stderr)
        return False

    return True


def run_benchmark():
    """Run bench/run.py and return the parsed JSON results."""
    print("  Benchmarking...", file=sys.stderr)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json_out = f.name

    try:
        out = subprocess.run(
            [sys.executable, str(BENCH_RUN),
             "--cold-iters", "0",
             "--iters", str(ITERS),
             "--json-out", json_out],
            capture_output=True, text=True,
        )
        if out.returncode != 0:
            print("  BENCHMARK FAILED:", file=sys.stderr)
            print(out.stdout, file=sys.stderr)
            print(out.stderr, file=sys.stderr)
            return None

        with open(json_out) as f:
            results = json.load(f)
        return results
    finally:
        if os.path.exists(json_out):
            os.unlink(json_out)


def threshold_label(t: int) -> str:
    """Human-readable label for the threshold."""
    if t == 0:
        return "always-heap"
    if t >= 999999:
        return "always-linear"
    return str(t)


def build_csv_data(results_list):
    """
    Build a list of (threshold_label, file, x8r_warm_min_ms, tik_warm_min_ms) rows.
    """
    rows = []
    for label, results in results_list:
        for fname in TARGET_FILES:
            if fname not in results:
                print(f"  WARNING: {fname} not in benchmark results", file=sys.stderr)
                continue
            fr = results[fname]
            if "x8r" not in fr or "tiktoken" not in fr:
                print(f"  WARNING: missing engine data for {fname}", file=sys.stderr)
                continue
            xw = fr["x8r"]["warm"]["min"]
            tw = fr["tiktoken"]["warm"]["min"]
            rows.append((label, fname, xw, tw))
    return rows


def write_csv(rows, path):
    """Write CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "file", "x8r_warm_min_ms", "tiktoken_warm_min_ms"])
        for row in rows:
            w.writerow(row)
    print(f"  Wrote {path}", file=sys.stderr)


def generate_chart(csv_path, png_path):
    """Generate a matplotlib chart from the CSV data."""
    import csv as csv_mod
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Read CSV
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            rows.append(row)

    thresh_order = ["always-heap", "16", "32", "64", "100", "200", "always-linear"]
    file_order = TARGET_FILES

    # Group by file
    file_data = {}
    for fname in file_order:
        file_data[fname] = {}

    for row in rows:
        thresh = row["threshold"]
        fname = row["file"]
        xw = float(row["x8r_warm_min_ms"])
        tw = float(row["tiktoken_warm_min_ms"])
        if fname not in file_data:
            file_data[fname] = {}
        file_data[fname][thresh] = {"x8r": xw, "tik": tw}

    # Find reasonable y-limit
    all_x8r = []
    all_tik = []
    for fname in file_order:
        for t in thresh_order:
            if t in file_data.get(fname, {}):
                all_x8r.append(file_data[fname][t]["x8r"])
                all_tik.append(file_data[fname][t]["tik"])

    fig, ax = plt.subplots(figsize=(12, 7))

    x_pos = range(len(thresh_order))
    markers = ["o", "s", "D", "^", "v", "<"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for fi, fname in enumerate(file_order):
        x8r_vals = []
        tik_vals = []
        for t in thresh_order:
            if t in file_data.get(fname, {}):
                x8r_vals.append(file_data[fname][t]["x8r"])
                tik_vals.append(file_data[fname][t]["tik"])
            else:
                x8r_vals.append(None)
                tik_vals.append(None)

        # Plot x8r line (solid)
        valid_x = [i for i, v in enumerate(x8r_vals) if v is not None]
        valid_y = [v for v in x8r_vals if v is not None]
        ax.plot(valid_x, valid_y, marker=markers[fi % len(markers)],
                color=colors[fi % len(colors)], linestyle="-",
                label=f"{fname} (x8r)", linewidth=2, markersize=6)

        # Plot tiktoken dashed reference for this file
        valid_y_t = [v for v in tik_vals if v is not None]
        if valid_y_t:
            # horizontal dashed line at tiktoken average for this file
            avg_tik = sum(tik_vals) / len(tik_vals)  # should be uniform across thresholds
            ax.axhline(y=avg_tik, color=colors[fi % len(colors)], linestyle=":",
                       alpha=0.5, linewidth=1.5)
            # label the tiktoken line with text
            ax.text(len(thresh_order) - 0.2, avg_tik, f"{fname} (tik)",
                    color=colors[fi % len(colors)], alpha=0.6, fontsize=8,
                    va="center", ha="left")

    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(thresh_order, rotation=30, ha="right")
    ax.set_xlabel("BPE_HEAP_THRESHOLD (pre-token byte length)", fontsize=12)
    ax.set_ylabel("Warm-min Runtime (ms)", fontsize=12)
    ax.set_title("x8r BPE Encoding: Heap Threshold Sweep", fontsize=14)
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=9, ncol=1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(png_path), dpi=150)
    plt.close(fig)
    print(f"  Wrote {png_path}", file=sys.stderr)


def main():
    print("=" * 70, file=sys.stderr)
    print(f"BPE_HEAP_THRESHOLD Sweep", file=sys.stderr)
    print(f"Thresholds: {THRESHOLDS}", file=sys.stderr)
    print(f"Target files: {TARGET_FILES}", file=sys.stderr)
    print(f"Warm iterations: {ITERS}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    all_rows = []

    for t in THRESHOLDS:
        label = threshold_label(t)

        rebuild_x8r(t)

        if not verify_correctness():
            print(f"  SKIPPING threshold {label}: correctness check failed", file=sys.stderr)
            continue

        results = run_benchmark()
        if results is None:
            print(f"  SKIPPING threshold {label}: benchmark failed", file=sys.stderr)
            continue

        rows = build_csv_data([(label, results)])
        all_rows.extend(rows)

        # Print per-file summary
        for r in rows:
            _, fname, xw, tw = r
            ratio = tw / xw if xw > 0 else 0
            speedup = f"{ratio:.2f}x {'(x8r faster)' if ratio > 1 else '(tik faster)'}"
            print(f"  {fname:35s} thresh={label:>14s}  x8r={xw:>9.3f}ms  tik={tw:>9.3f}ms  {speedup}", file=sys.stderr)

    if not all_rows:
        print("No valid data collected!", file=sys.stderr)
        sys.exit(1)

    write_csv(all_rows, CSV_PATH)

    # Generate chart using altair
    generate_chart(CSV_PATH, PNG_PATH)

    print("\n=== Sweep complete ===", file=sys.stderr)
    print(f"CSV: {CSV_PATH}", file=sys.stderr)
    print(f"PNG: {PNG_PATH}", file=sys.stderr)
    print(file=sys.stderr)

    # Also print the CSV to stdout for the report
    print("\n# CSV Data")
    print("# " + CSV_PATH.read_text())


if __name__ == "__main__":
    main()
