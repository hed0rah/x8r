#!/usr/bin/env python3
"""
generate a varied benchmark corpus under bench/corpus/.

goal: reproducible inputs that exercise different tokenizer paths
(ascii code, utf-8 prose, markdown, json, mixed) at multiple sizes,
plus stressor files designed to produce >32-byte pre-tokens for the
BPE merge-loop threshold split.
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

# CJK Unified Ideographs range for stressor generation
CJK_START = 0x4E00
CJK_END   = 0x9FFF
CJK_COUNT = CJK_END - CJK_START + 1


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
        out.append(line)
        total += len(line.encode("utf-8"))
    s = "".join(out)
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


# ---------------------------------------------------------------------------
# Stressor generators -- designed to produce pre-tokens >32 bytes
# ---------------------------------------------------------------------------

STRESSOR_TARGET = 128 * 1024  # 128 KB minimum for stable timing


def gen_cjk_dense(target_bytes: int) -> str:
    """Continuous CJK paragraphs with no ASCII spaces.

    Each paragraph (separated by a newline) is one giant \p{L}+ pre-token.
    CJK characters are 'Letter, Other' in Unicode and match \p{L} in the
    cl100k/o200k pre-token regex.  Each CJK codepoint is 3 bytes in UTF-8,
    so even a modest paragraph easily exceeds 32 bytes.
    """
    out = []
    total = 0
    while total < target_bytes:
        para_len = random.randint(200, 800)  # codepoints per paragraph
        para = "".join(chr(random.randint(CJK_START, CJK_END)) for _ in range(para_len))
        block = para + "\n"
        out.append(block)
        total += len(block.encode("utf-8"))
    result = "".join(out)
    b = result.encode("utf-8")[:target_bytes]
    while True:
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            b = b[:-1]


def gen_long_words(target_bytes: int) -> str:
    """Latin pseudo-words 50-200 letters long, separated by single spaces.

    Each word is one long \p{L}+ pre-token.  Words are purely alphabetic
    (no digits, no punct) so the regex stays inside the \p{L}+ alternative
    for the full length of the word.
    """
    vowels = "aeiou"
    consonants = "bcdfghjklmnpqrstvwxyz"
    letters = vowels + consonants
    out = []
    total = 0
    first = True
    while total < target_bytes:
        if not first:
            out.append(" ")
            total += 1
        first = False
        word_len = random.randint(50, 200)
        word = "".join(random.choice(letters) for _ in range(word_len))
        out.append(word)
        total += word_len
    return "".join(out)[:target_bytes]


def gen_deep_indent(target_bytes: int) -> str:
    """Code-shaped lines with 8-16 levels of nesting using 4-space indents.

    Each indent run is a contiguous \s+ pre-token.  At 8 levels (32 spaces)
    we are at the threshold boundary; at 16 levels (64 spaces) we are
    comfortably in heap territory.
    """
    code_fragments = [
        "int x = 1;",
        "if (x) {",
        "for (int i = 0; i < n; i++) {",
        "while (cond) {",
        "switch (x) {",
        "case 0:",
        "return x + 1;",
        "break;",
        "continue;",
        "x = x * 2;",
        "result = compute(x);",
        "output(result);",
    ]
    out = []
    total = 0
    while total < target_bytes:
        level = random.randint(8, 16)
        indent = "    " * level
        fragment = random.choice(code_fragments)
        line = indent + fragment + "\n"
        out.append(line)
        total += len(line)
    return "".join(out)[:target_bytes]


def gen_markdown_rules(target_bytes: int) -> str:
    """Markdown with long horizontal rules and walls of repeated punct chars.

    Long runs of ===, ---, ////, ####, etc match the
    [^\s\p{L}\p{N}]+ alternative and produce long pre-tokens.
    """
    punct_runs = []
    for ch in ["-", "=", "/", "#", "*", "~", "_", "+"]:
        punct_runs.append(ch)

    out = []
    total = 0
    while total < target_bytes:
        roll = random.random()
        if roll < 0.35:
            # Long horizontal rule
            ch = random.choice(["-", "=", "*", "~"])
            line = ch * random.randint(60, 120) + "\n"
            out.append(line)
            total += len(line)
        elif roll < 0.60:
            # Punct wall without newline (continues on same line)
            ch = random.choice(punct_runs)
            run = ch * random.randint(40, 80)
            out.append(run)
            total += len(run)
        else:
            # Short marker line to break up the pattern
            line = random.choice(["# Section\n", "> note\n", "---\n", "***\n"])
            out.append(line)
            total += len(line)

    # Trim to exact target
    result = "".join(out)
    # Make sure we don't cut mid-run; trim at last \n boundary
    result = result[:target_bytes]
    last_nl = result.rfind("\n")
    if last_nl > 0:
        result = result[:last_nl + 1]
    return result


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

STRESSORS = {
    "cjk_dense":         gen_cjk_dense,
    "long_words":        gen_long_words,
    "deep_indent":       gen_deep_indent,
    "markdown_rules":    gen_markdown_rules,
}


def main():
    for kind, fn in GENERATORS.items():
        for size_name, nbytes in SIZES.items():
            path = OUT / f"{kind}_{size_name}.txt"
            data = fn(nbytes)
            path.write_text(data, encoding="utf-8")
            print(f"{path.name}: {path.stat().st_size} bytes")

    # Stressor files -- single size, ~128 KB each
    for kind, fn in STRESSORS.items():
        path = OUT / f"{kind}.txt"
        data = fn(STRESSOR_TARGET)
        path.write_text(data, encoding="utf-8")
        actual = path.stat().st_size
        print(f"{path.name}: {actual} bytes [stressor, target={STRESSOR_TARGET}]")


if __name__ == "__main__":
    main()
