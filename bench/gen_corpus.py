#!/usr/bin/env python3
"""
generate a varied benchmark corpus under bench/corpus/.

goal: reproducible inputs that exercise different tokenizer paths
(ascii code, utf-8 prose, markdown, json, mixed) at multiple sizes.
"""
import os
import random
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "corpus"
OUT.mkdir(exist_ok=True)

random.seed(0xC0FFEE)

ASCII_CODE_VOCAB = [
    "static", "int", "void", "const", "char", "if", "else", "for", "while",
    "return", "struct", "typedef", "size_t", "uint8_t", "NULL", "sizeof",
    "printf", "malloc", "free", "memcpy", "memset", "assert", "goto",
    "break", "continue", "switch", "case", "default", "inline", "extern",
]
IDENT_CHARS = string.ascii_letters + "_"
IDENT_TAIL = string.ascii_letters + string.digits + "_"

UTF8_PROSE_WORDS = [
    "résumé", "naïve", "café", "façade", "jalapeño", "fiancée",
    "élite", "überleben", "schön", "Mädchen", "αλφα", "βητα",
    "привет", "мир", "日本語", "中文", "한국어", "emoji", "—",
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
]

def gen_ascii_code(target_bytes: int) -> str:
    out = []
    total = 0
    while total < target_bytes:
        name = random.choice(IDENT_CHARS) + "".join(random.choices(IDENT_TAIL, k=random.randint(2, 12)))
        kw = random.choice(ASCII_CODE_VOCAB)
        n = random.randint(0, 999)
        line = f"{kw} {name} = {n}; /* {random.choice(ASCII_CODE_VOCAB)} */\n"
        if random.random() < 0.15:
            line = f"\nvoid {name}(int x) {{\n    return x * {n};\n}}\n\n"
        out.append(line)
        total += len(line)
    return "".join(out)[:target_bytes]

def gen_utf8_prose(target_bytes: int) -> str:
    out = []
    total = 0
    while total < target_bytes:
        words = [random.choice(UTF8_PROSE_WORDS) for _ in range(random.randint(6, 14))]
        line = " ".join(words) + ".\n"
        b = line.encode("utf-8")
        out.append(line)
        total += len(b)
    s = "".join(out)
    # trim at a codepoint boundary
    b = s.encode("utf-8")[:target_bytes]
    while True:
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            b = b[:-1]

def gen_markdown(target_bytes: int) -> str:
    out = []
    total = 0
    h = 1
    while total < target_bytes:
        if random.random() < 0.1:
            line = f"\n{'#' * random.randint(1, 4)} Section {h}\n\n"
            h += 1
        elif random.random() < 0.2:
            line = "```\n" + gen_ascii_code(random.randint(80, 400)) + "```\n"
        elif random.random() < 0.3:
            line = f"- item {random.randint(1,99)}: {random.choice(UTF8_PROSE_WORDS)}\n"
        else:
            words = [random.choice(UTF8_PROSE_WORDS + ASCII_CODE_VOCAB) for _ in range(random.randint(8, 20))]
            line = " ".join(words) + ".\n"
        out.append(line)
        total += len(line.encode("utf-8"))
    s = "".join(out)
    b = s.encode("utf-8")[:target_bytes]
    while True:
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            b = b[:-1]

def gen_json(target_bytes: int) -> str:
    import json
    out = []
    total = 0
    arr = []
    while total < target_bytes:
        obj = {
            "id": random.randint(0, 10**9),
            "name": random.choice(UTF8_PROSE_WORDS),
            "tags": random.sample(ASCII_CODE_VOCAB, k=random.randint(1, 5)),
            "score": random.random(),
            "active": random.choice([True, False, None]),
        }
        arr.append(obj)
        total += len(json.dumps(obj))
        if len(arr) > 10000:
            break
    return json.dumps(arr, indent=2, ensure_ascii=False)[:target_bytes]

SIZES = {
    "small":  8 * 1024,
    "medium": 128 * 1024,
    "large":  1024 * 1024,
}

GENERATORS = {
    "ascii_code": gen_ascii_code,
    "utf8_prose": gen_utf8_prose,
    "markdown":   gen_markdown,
    "json":       gen_json,
}

def main():
    for kind, fn in GENERATORS.items():
        for size_name, nbytes in SIZES.items():
            path = OUT / f"{kind}_{size_name}.txt"
            data = fn(nbytes)
            path.write_text(data, encoding="utf-8")
            print(f"{path.name}: {path.stat().st_size} bytes")

if __name__ == "__main__":
    main()
