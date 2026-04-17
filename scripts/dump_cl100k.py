#!/usr/bin/env python3
"""
dump tiktoken cl100k_base vocab into x8r's flat vocab format.

format (little-endian):
  magic        "X8RV"
  version      u32 = 1
  vocab_id     u32 = 0 (cl100k)
  n_tokens     u32
  table_size   u32 (power of 2)
  data_bytes   u32
  reserved     u32 = 0
  table[table_size]  u32  offset into data blob, or 0xFFFFFFFF for empty
  data[...]          u8   packed (u16 len | u32 rank | bytes[len])

hash: fnv-1a 32-bit over the token bytes. must match src/internal.h.
open addressing, linear probing.
"""

import struct
import sys
from pathlib import Path


def fnv1a(data: bytes) -> int:
    h = 2166136261
    for b in data:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def next_pow2_for_load(n: int, load: float = 0.5) -> int:
    want = int(n / load) + 1
    s = 1
    while s < want:
        s <<= 1
    return s


def main(out_path: str):
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    # mergeable_ranks: dict[bytes, int]
    ranks = enc._mergeable_ranks
    # tiktoken also has a handful of "special" tokens; we skip those for v1
    # since they never appear via the pretokenizer path.
    items = sorted(ranks.items(), key=lambda kv: kv[1])

    n = len(items)
    table_size = next_pow2_for_load(n, load=0.5)
    mask = table_size - 1
    print(f"tokens={n} table_size={table_size} load={n/table_size:.3f}", file=sys.stderr)

    # build data blob
    data = bytearray()
    entry_off = []  # index by token index
    for tok, rank in items:
        off = len(data)
        if len(tok) > 0xFFFF:
            raise RuntimeError(f"token too long: {len(tok)}")
        data += struct.pack("<HI", len(tok), rank)
        data += tok
        entry_off.append(off)

    if len(data) >= 0xFFFFFFFF:
        raise RuntimeError("data blob too big for u32 offsets")

    # build hash table
    EMPTY = 0xFFFFFFFF
    table = [EMPTY] * table_size
    for (tok, rank), off in zip(items, entry_off):
        h = fnv1a(tok) & mask
        while table[h] != EMPTY:
            h = (h + 1) & mask
        table[h] = off

    header = b"X8RV" + struct.pack("<IIIIII", 1, 0, n, table_size, len(data), 0)

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(struct.pack(f"<{table_size}I", *table))
        f.write(data)

    size_mb = (len(header) + 4 * table_size + len(data)) / (1024 * 1024)
    print(f"wrote {out_path} ({size_mb:.1f} MiB)", file=sys.stderr)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "vocab/cl100k.bin"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    main(out)
