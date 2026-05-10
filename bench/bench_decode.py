#!/usr/bin/env python3
"""benchmark x8r --decode vs tiktoken.Encoding.decode_bytes.

both consume a token-id list and produce raw bytes. the comparison is
not perfectly apples-to-apples because tiktoken runs in-process while
x8r runs as a subprocess (paying fork+exec+stdin-parse on every run).
that's how a typical caller would actually invoke x8r from python, so
the wall-clock comparison is what users would observe.

reports warm-min ms across iters runs per file, plus derived
throughput in MB/s of decoded output.
"""
import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parent.parent
X8R = str(ROOT / "build" / "x8r")
CORPUS = ROOT / "bench" / "corpus"

MODEL = os.environ.get("X8R_BENCH_MODEL", "cl100k")
if MODEL not in ("cl100k", "o200k"):
    raise SystemExit(f"X8R_BENCH_MODEL must be cl100k or o200k, got {MODEL!r}")
VOCAB = str(ROOT / "vocab" / f"{MODEL}.bin")
ENC = tiktoken.get_encoding(f"{MODEL}_base")

print(f"benchmarking decode: x8r vs tiktoken {MODEL}_base", file=sys.stderr)


def time_x8r_decode(ids_input: bytes) -> float:
    env = dict(os.environ, X8R_VOCAB=VOCAB)
    t0 = time.perf_counter_ns()
    r = subprocess.run([X8R, "--decode", "-"], input=ids_input,
                       capture_output=True, env=env, check=True)
    t1 = time.perf_counter_ns()
    return (t1 - t0) / 1e6, r.stdout


def time_tik_decode(ids: list[int]) -> float:
    t0 = time.perf_counter_ns()
    out = ENC.decode_bytes(ids)
    t1 = time.perf_counter_ns()
    return (t1 - t0) / 1e6, out


def stats(samples: list[float]) -> dict:
    if not samples:
        return {}
    return {
        "n": len(samples),
        "min": min(samples),
        "median": st.median(samples),
        "mean": st.mean(samples),
        "stddev": st.stdev(samples) if len(samples) > 1 else 0.0,
        "max": max(samples),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--json-out", default=str(ROOT / "bench" / "decode_results.json"))
    ap.add_argument("--only", help="glob filter on corpus filenames")
    args = ap.parse_args()

    if not Path(X8R).exists():
        print(f"missing {X8R}; run 'make' first", file=sys.stderr)
        sys.exit(2)

    files = sorted(CORPUS.glob("*.txt"))
    if args.only:
        import fnmatch
        files = [f for f in files if fnmatch.fnmatch(f.name, args.only)]

    results = {}
    print(f"\n{'file':32s} {'ids':>8s} {'bytes':>9s}  {'x8r_warm':>10s} {'tik_warm':>10s} {'speedup':>8s}  {'x8r_MB/s':>9s} {'tik_MB/s':>9s}")
    print("-" * 130)
    for f in files:
        # encode once via tiktoken, use the resulting id list as decode input
        text = f.read_bytes().decode("utf-8", errors="replace")
        ids = ENC.encode_ordinary(text)
        ids_input = ("\n".join(str(i) for i in ids) + "\n").encode("ascii")
        n_ids = len(ids)

        # warm both paths
        time_x8r_decode(ids_input)
        time_tik_decode(ids)

        # measure
        x8r_samples = []
        tik_samples = []
        x8r_bytes = b""
        tik_bytes = b""
        for _ in range(args.iters):
            ms, x8r_bytes = time_x8r_decode(ids_input)
            x8r_samples.append(ms)
            ms, tik_bytes = time_tik_decode(ids)
            tik_samples.append(ms)

        # correctness gate inside the bench
        if x8r_bytes != tik_bytes:
            print(f"  MISMATCH on {f.name}: x8r={len(x8r_bytes)}B tik={len(tik_bytes)}B", file=sys.stderr)
            sys.exit(1)

        n_bytes = len(x8r_bytes)
        x8r_min = min(x8r_samples)
        tik_min = min(tik_samples)
        x8r_mbs = n_bytes / 1e6 / (x8r_min / 1e3)
        tik_mbs = n_bytes / 1e6 / (tik_min / 1e3)
        speedup = tik_min / x8r_min if x8r_min > 0 else 0.0

        results[f.name] = {
            "n_ids": n_ids,
            "n_bytes": n_bytes,
            "x8r": stats(x8r_samples),
            "tiktoken": stats(tik_samples),
        }
        print(f"{f.name:32s} {n_ids:>8d} {n_bytes:>9d}  "
              f"{x8r_min:>9.2f}ms {tik_min:>9.2f}ms {speedup:>7.2f}x  "
              f"{x8r_mbs:>8.1f} {tik_mbs:>8.1f}")

    Path(args.json_out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
