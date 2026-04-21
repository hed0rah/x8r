#include "internal.h"
#include "unicode_tables.h"

#include <stddef.h>
#include <stdint.h>

/*
 * scalar pre-tokenizer matching the o200k regex.
 *
 *   [^\r\n\p{L}\p{N}]? [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]* [\p{Ll}\p{Lm}\p{Lo}\p{M}]+ (?i:'s|'t|'re|'ve|'m|'ll|'d)?
 * | [^\r\n\p{L}\p{N}]? [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+ [\p{Ll}\p{Lm}\p{Lo}\p{M}]* (?i:'s|'t|'re|'ve|'m|'ll|'d)?
 * | \p{N}{1,3}
 * |  ?[^\s\p{L}\p{N}]+[\r\n/]*
 * | \s*[\r\n]+
 * | \s+(?!\S)
 * | \s+
 *
 * differences vs cl100k:
 *   - contractions are a suffix on letter runs, not a standalone rule
 *   - letter runs come in two shapes (CamelCase split):
 *       "upperish* lowerish+" is tried first, then "upperish+ lowerish*"
 *     upperish = Lu|Lt|Lm|Lo|M,  lowerish = Ll|Lm|Lo|M
 *     (Lm, Lo, M are neutral - they match both classes)
 *   - punct runs absorb trailing '/' in addition to \r\n
 *
 * regex alternation is first-match-wins (fancy-regex semantics, same as
 * tiktoken). within each alt, * and + are greedy with backtracking.
 *
 * known divergence: for unassigned codepoints (unicode category Cn) that
 * sit inside a script-allocated block, x8r uses python's `regex` module
 * classification (cp counts as \p{L}). tiktoken's fancy-regex crate does
 * not, so on random noise containing such codepoints x8r and tiktoken's
 * public encode_ordinary() will rarely disagree. tiktoken's own internal
 * _encode_only_native_bpe() path agrees with x8r. real-world text and
 * source code never hit this.
 */

#define CLS_LETTER  X8R_CLS_LETTER
#define CLS_DIGIT   X8R_CLS_DIGIT
#define CLS_SPACE   X8R_CLS_SPACE
#define CLS_NEWLINE X8R_CLS_NEWLINE
#define CLS_UPPER   X8R_CLS_UPPER
#define CLS_LOWER   X8R_CLS_LOWER
#define CLS_MARK    X8R_CLS_MARK

typedef struct { uint32_t cls; uint32_t blen; } cp_info;

/* decode one utf-8 codepoint at buf[p]. returns class bits and byte
 * length. on invalid utf-8 we return blen=1 and cls=0, so the top loop
 * emits the bad byte as its own pre-token. */
static inline cp_info cp_at(const uint8_t *b, size_t n, size_t p) {
    cp_info r = {0, 1};
    uint8_t c = b[p];
    uint32_t cp;
    if (c < 0x80) {
        cp = c;
    } else if ((c & 0xE0) == 0xC0 && p + 1 < n && (b[p+1] & 0xC0) == 0x80) {
        cp = ((uint32_t)(c & 0x1F) << 6) | (b[p+1] & 0x3F);
        if (cp < 0x80) return r;
        r.blen = 2;
    } else if ((c & 0xF0) == 0xE0 && p + 2 < n && (b[p+1] & 0xC0) == 0x80 && (b[p+2] & 0xC0) == 0x80) {
        cp = ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(b[p+1] & 0x3F) << 6) | (b[p+2] & 0x3F);
        if (cp < 0x800) return r;
        if (cp >= 0xD800 && cp <= 0xDFFF) return r;
        r.blen = 3;
    } else if ((c & 0xF8) == 0xF0 && p + 3 < n
               && (b[p+1] & 0xC0) == 0x80 && (b[p+2] & 0xC0) == 0x80 && (b[p+3] & 0xC0) == 0x80) {
        cp = ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(b[p+1] & 0x3F) << 12)
           | ((uint32_t)(b[p+2] & 0x3F) << 6) | (b[p+3] & 0x3F);
        if (cp < 0x10000 || cp > 0x10FFFF) return r;
        r.blen = 4;
    } else {
        return r;
    }
    r.cls = x8r_cp_class(cp);
    return r;
}

static inline int has_any(uint32_t cls, uint32_t mask) { return (cls & mask) != 0; }

/* upperish = [Lu Lt Lm Lo M]  = LETTER_without_LOWER, or MARK
 * lowerish = [Ll Lm Lo M]     = LETTER_without_UPPER, or MARK */
static inline int is_upperish(uint32_t cls) {
    if (cls & CLS_MARK) return 1;
    return (cls & CLS_LETTER) && !(cls & CLS_LOWER);
}
static inline int is_lowerish(uint32_t cls) {
    if (cls & CLS_MARK) return 1;
    return (cls & CLS_LETTER) && !(cls & CLS_UPPER);
}

/* walk back from q (byte offset) to the start of the previous codepoint.
 * valid utf-8 or 1-byte invalid bytes: skip continuation bytes. */
static inline size_t prev_cp_start(const uint8_t *b, size_t lo, size_t q) {
    if (q <= lo) return lo;
    size_t r = q - 1;
    while (r > lo && (b[r] & 0xC0) == 0x80) r--;
    return r;
}

/* (?i:'s|'t|'re|'ve|'m|'ll|'d) - case-insensitive ASCII only */
static size_t try_contraction(const uint8_t *b, size_t n, size_t p) {
    if (p >= n || b[p] != '\'') return 0;
    if (p + 1 >= n) return 0;
    uint8_t c1 = b[p + 1]; if (c1 >= 'A' && c1 <= 'Z') c1 += 32;
    if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return 2;
    if (p + 2 >= n) return 0;
    uint8_t c2 = b[p + 2]; if (c2 >= 'A' && c2 <= 'Z') c2 += 32;
    if (c1 == 'r' && c2 == 'e') return 3;
    if (c1 == 'v' && c2 == 'e') return 3;
    if (c1 == 'l' && c2 == 'l') return 3;
    return 0;
}

/* match "upperish* lowerish+" starting at p. returns bytes consumed or 0.
 *
 * greedy upperish* first, then lowerish+ must match at least one cp.
 * if lowerish+ cannot start at the end of the upperish run (next cp is
 * not lowerish), we backtrack one cp at a time: only neutral cps (Lm, Lo,
 * M) can be given back to lowerish+ since they match both classes. */
static size_t match_upperstar_lowerplus(const uint8_t *b, size_t n, size_t p) {
    /* greedy upperish* */
    size_t q = p;
    while (q < n) {
        cp_info c = cp_at(b, n, q);
        if (!is_upperish(c.cls)) break;
        q += c.blen;
    }
    /* try lowerish+ with backtracking */
    for (;;) {
        if (q < n) {
            cp_info c = cp_at(b, n, q);
            if (is_lowerish(c.cls)) {
                size_t r = q + c.blen;
                while (r < n) {
                    cp_info cc = cp_at(b, n, r);
                    if (!is_lowerish(cc.cls)) break;
                    r += cc.blen;
                }
                return r - p;
            }
        }
        /* need to backtrack: give the last consumed upperish cp to lowerish+ */
        if (q <= p) return 0;
        size_t prev = prev_cp_start(b, p, q);
        cp_info pc = cp_at(b, n, prev);
        if (!is_lowerish(pc.cls)) return 0;
        q = prev;
    }
}

/* match "upperish+ lowerish*". greedy upperish+, then greedy lowerish*.
 * no backtracking needed: if upperish+ consumed neutral cps at the end,
 * lowerish* still matches them... but lowerish* is after upperish+ in
 * position so those cps are already consumed. that's fine - lowerish*
 * can match 0 chars.
 *
 * subtle point: if upperish+ greedily consumes a trailing neutral that
 * a human would read as "belonging to the lower part," the full match
 * length is unchanged (cp is either upperish or lowerish, just attributed
 * differently). we return total bytes, so it doesn't matter. */
static size_t match_upperplus_lowerstar(const uint8_t *b, size_t n, size_t p) {
    size_t q = p;
    while (q < n) {
        cp_info c = cp_at(b, n, q);
        if (!is_upperish(c.cls)) break;
        q += c.blen;
    }
    if (q == p) return 0;
    while (q < n) {
        cp_info c = cp_at(b, n, q);
        if (!is_lowerish(c.cls)) break;
        q += c.blen;
    }
    return q - p;
}

/* try a letter rule with optional leader and optional trailing contraction.
 * body_fn is either match_upperstar_lowerplus or match_upperplus_lowerstar. */
static size_t try_letter_rule(const uint8_t *b, size_t n, size_t p,
                              size_t (*body_fn)(const uint8_t *, size_t, size_t)) {
    /* leader: [^\r\n\p{L}\p{N}]? - try with leader first (greedy ?) */
    cp_info lead = cp_at(b, n, p);
    int leader_ok = lead.blen > 0 && p + lead.blen < n
                    && !has_any(lead.cls, CLS_LETTER | CLS_DIGIT | CLS_NEWLINE);
    if (leader_ok) {
        size_t body = body_fn(b, n, p + lead.blen);
        if (body > 0) {
            size_t total = lead.blen + body;
            size_t contr = try_contraction(b, n, p + total);
            return total + contr;
        }
    }
    /* fall through: without leader */
    size_t body = body_fn(b, n, p);
    if (body > 0) {
        size_t contr = try_contraction(b, n, p + body);
        return body + contr;
    }
    return 0;
}

static size_t try_letter_run(const uint8_t *b, size_t n, size_t p) {
    /* alt 1 is tried first */
    size_t m = try_letter_rule(b, n, p, match_upperstar_lowerplus);
    if (m > 0) return m;
    return try_letter_rule(b, n, p, match_upperplus_lowerstar);
}

/* \p{N}{1,3} */
static size_t try_digits(const uint8_t *b, size_t n, size_t p) {
    size_t r = p;
    int count = 0;
    while (r < n && count < 3) {
        cp_info c = cp_at(b, n, r);
        if (!(c.cls & CLS_DIGIT)) break;
        r += c.blen;
        count++;
    }
    return r - p;
}

/*  ?[^\s\p{L}\p{N}]+[\r\n/]*   - trailing class includes '/' in o200k */
static size_t try_punct_run(const uint8_t *b, size_t n, size_t p) {
    size_t q = p;
    if (q < n && b[q] == ' ') q++;
    size_t punct_start = q;
    while (q < n) {
        cp_info c = cp_at(b, n, q);
        if (has_any(c.cls, CLS_SPACE | CLS_LETTER | CLS_DIGIT)) break;
        q += c.blen;
    }
    if (q == punct_start) return 0;
    while (q < n && (b[q] == '\r' || b[q] == '\n' || b[q] == '/')) q++;
    return q - p;
}

/* \s*[\r\n]+  - implemented by taking \s+ greedy then shrinking back to
 * end on \r or \n. equivalent to the backtracking the regex would do. */
static size_t try_space_then_newline(const uint8_t *b, size_t n, size_t p) {
    size_t q = p;
    for (;;) {
        q = x8r_scan_ascii_space(b, n, q);
        if (q >= n) break;
        cp_info c = cp_at(b, n, q);
        if (!(c.cls & CLS_SPACE)) break;
        q += c.blen;
    }
    size_t r = q;
    while (r > p && b[r - 1] != '\r' && b[r - 1] != '\n') r--;
    if (r == p) return 0;
    return r - p;
}

/* \s+(?!\S) - whitespace run that ends at EOF or right before more ws
 * (i.e., the run where subsequent input is ws/EOF, not non-ws). */
static size_t try_trailing_ws(const uint8_t *b, size_t n, size_t p) {
    size_t q = p;
    for (;;) {
        q = x8r_scan_ascii_space(b, n, q);
        if (q >= n) break;
        cp_info c = cp_at(b, n, q);
        if (!(c.cls & CLS_SPACE)) break;
        q += c.blen;
    }
    if (q == p) return 0;
    if (q == n) return q - p;
    /* leave one codepoint of ws behind */
    size_t last = p;
    size_t walk = p;
    while (walk < q) {
        last = walk;
        cp_info c = cp_at(b, n, walk);
        walk += c.blen;
    }
    if (last == p) return 0;
    return last - p;
}

/* \s+ - any whitespace run */
static size_t try_any_ws(const uint8_t *b, size_t n, size_t p) {
    size_t q = p;
    for (;;) {
        q = x8r_scan_ascii_space(b, n, q);
        if (q >= n) break;
        cp_info c = cp_at(b, n, q);
        if (!(c.cls & CLS_SPACE)) break;
        q += c.blen;
    }
    return q - p;
}

size_t x8r_pretokenize_o200k(const uint8_t *buf, size_t len,
                             x8r_pretok_sink sink, void *user) {
    size_t p = 0;
    size_t count = 0;
    while (p < len) {
        size_t take = 0;

        if ((take = try_letter_run(buf, len, p))) goto emit;
        if ((take = try_digits(buf, len, p))) goto emit;
        if ((take = try_punct_run(buf, len, p))) goto emit;
        if ((take = try_space_then_newline(buf, len, p))) goto emit;
        if ((take = try_trailing_ws(buf, len, p))) goto emit;
        if ((take = try_any_ws(buf, len, p))) goto emit;

        { cp_info c = cp_at(buf, len, p); take = c.blen; }

    emit:
        sink(user, p, take);
        p += take;
        count++;
    }
    return count;
}
