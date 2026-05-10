#!/usr/bin/env python3
"""in-process decode benchmark: libx8r.so via ctypes vs tiktoken.

this is the apples-to-apples measurement. the subprocess CLI bench is
dominated by fork+exec+stdin-parse overhead and is not representative
of how decode would be used in a real integration.

both paths run in the same python process. x8r decode goes through
libx8r.so's x8r_decode_bytes via a tiny ctypes wrapper. tiktoken
goes through its native rust decode_bytes.
"""
import argparse
import ctypes as C
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "build" / "libx8r.so"
CORPUS = ROOT / "bench" / "corpus"

MODEL = os.environ.get("X8R_BENCH_MODEL", "cl100k")
if MODEL not in ("cl100k", "o200k"):
    raise SystemExit(f"X8R_BENCH_MODEL must be cl100k or o200k, got {MODEL!r}")
VOCAB = str(ROOT / "vocab" / f"{MODEL}.bin")
ENC = tiktoken.get_encoding(f"{MODEL}_base")

print(f"in-process decode bench: libx8r.so vs tiktoken {MODEL}_base", file=sys.stderr)

# ---- ctypes bindings ----------------------------------------------------
lib = C.CDLL(str(LIB))

X8R_VOCAB_AUTO = 0xFFFF

# x8r_status x8r_ctx_open(const char *vocab_path, x8r_vocab_id vocab, x8r_ctx **out);
lib.x8r_ctx_open.restype = C.c_int
lib.x8r_ctx_open.argtypes = [C.c_char_p, C.c_uint32, C.POINTER(C.c_void_p)]
# void x8r_ctx_close(x8r_ctx *ctx);
lib.x8r_ctx_close.restype = None
lib.x8r_ctx_close.argtypes = [C.c_void_p]
# x8r_status x8r_decode_bytes(ctx, ids, n, &out_bytes, &out_len);
lib.x8r_decode_bytes.restype = C.c_int
lib.x8r_decode_bytes.argtypes = [
    C.c_void_p,
    C.POINTER(C.c_uint32),
    C.c_size_t,
    C.POINTER(C.POINTER(C.c_uint8)),
    C.POINTER(C.c_size_t),
]
# void x8r_bytes_free(uint8_t *bytes);
lib.x8r_bytes_free.restype = None
lib.x8r_bytes_free.argtypes = [C.c_void_p]


def open_ctx(vocab_path):
    ctx = C.c_void_p()
    rc = lib.x8r_ctx_open(vocab_path.encode(), X8R_VOCAB_AUTO, C.byref(ctx))
    if rc != 0:
        raise SystemExit(f"x8r_ctx_open failed: {rc}")
    return ctx


def x8r_decode_buf(ctx, arr_ptr, n_ids: int) -> bytes:
    """time the actual decode call; ids must already be a c_uint32 array."""
    out_p = C.POINTER(C.c_uint8)()
    out_n = C.c_size_t()
    rc = lib.x8r_decode_bytes(ctx, arr_ptr, n_ids, C.byref(out_p), C.byref(out_n))
    if rc != 0:
        raise RuntimeError(f"decode failed: {rc}")
    if out_n.value == 0:
        return b""
    buf = C.string_at(out_p, out_n.value)
    lib.x8r_bytes_free(out_p)
    return buf


def make_ctypes_arr(ids: list[int]):
    """build a c_uint32 array once. excluded from decode timing because
    tiktoken's binding accepts a python list directly via PyO3 and doesn't
    pay an explicit marshalling step."""
    return (C.c_uint32 * len(ids))(*ids)


# ---- bench --------------------------------------------------------------
def stats(samples):
    return {
        "n": len(samples),
        "min": min(samples),
        "median": st.median(samples),
        "stddev": st.stdev(samples) if len(samples) > 1 else 0.0,
        "max": max(samples),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=11)
    ap.add_argument("--json-out", default=str(ROOT / "bench" / "decode_inproc_results.json"))
    ap.add_argument("--only", help="glob filter on corpus filenames")
    args = ap.parse_args()

    if not LIB.exists():
        print(f"missing {LIB}; run 'make' first", file=sys.stderr); sys.exit(2)

    files = sorted(CORPUS.glob("*.txt"))
    if args.only:
        import fnmatch
        files = [f for f in files if fnmatch.fnmatch(f.name, args.only)]

    ctx = open_ctx(VOCAB)

    results = {}
    print(f"\n{'file':32s} {'ids':>8s} {'bytes':>9s}  {'x8r_warm':>10s} {'tik_warm':>10s} {'speedup':>8s}  {'x8r_MB/s':>9s} {'tik_MB/s':>9s}")
    print("-" * 130)
    for f in files:
        text = f.read_bytes().decode("utf-8", errors="replace")
        ids = ENC.encode_ordinary(text)
        n_ids = len(ids)

        # marshalling out of timing loop
        arr = make_ctypes_arr(ids)
        n_ids = len(ids)

        # warm
        x8r_decode_buf(ctx, arr, n_ids)
        ENC.decode_bytes(ids)

        x8r_samples = []
        tik_samples = []
        x8r_bytes = b""
        tik_bytes = b""
        for _ in range(args.iters):
            t0 = time.perf_counter_ns()
            x8r_bytes = x8r_decode_buf(ctx, arr, n_ids)
            t1 = time.perf_counter_ns()
            x8r_samples.append((t1 - t0) / 1e6)

            t0 = time.perf_counter_ns()
            tik_bytes = ENC.decode_bytes(ids)
            t1 = time.perf_counter_ns()
            tik_samples.append((t1 - t0) / 1e6)

        if x8r_bytes != tik_bytes:
            print(f"  MISMATCH on {f.name}", file=sys.stderr)
            sys.exit(1)

        n_bytes = len(x8r_bytes)
        x8r_min = min(x8r_samples)
        tik_min = min(tik_samples)
        x8r_mbs = n_bytes / 1e6 / (x8r_min / 1e3) if x8r_min > 0 else 0.0
        tik_mbs = n_bytes / 1e6 / (tik_min / 1e3) if tik_min > 0 else 0.0
        speedup = tik_min / x8r_min if x8r_min > 0 else 0.0

        results[f.name] = {
            "n_ids": n_ids,
            "n_bytes": n_bytes,
            "x8r": stats(x8r_samples),
            "tiktoken": stats(tik_samples),
        }
        print(f"{f.name:32s} {n_ids:>8d} {n_bytes:>9d}  "
              f"{x8r_min:>9.3f}ms {tik_min:>9.3f}ms {speedup:>7.2f}x  "
              f"{x8r_mbs:>8.1f} {tik_mbs:>8.1f}")

    lib.x8r_ctx_close(ctx)
    Path(args.json_out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
