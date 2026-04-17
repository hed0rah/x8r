#include "internal.h"
#include "unicode_tables.h"

#include <stddef.h>
#include <stdint.h>

/*
 * scalar pre-tokenizer matching the cl100k regex. operates on unicode
 * codepoints decoded from utf-8. classification is driven by the
 * two-stage table in unicode_tables.h (generated from python's
 * unicodedata.category).
 *
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)
 *   | [^\r\n\p{L}\p{N}]? \p{L}+
 *   | \p{N}{1,3}
 *   |  ?[^\s\p{L}\p{N}]+[\r\n]*
 *   | \s*[\r\n]+
 *   | \s+(?!\S)
 *   | \s+
 *
 * on invalid utf-8 we emit the bad byte as a 1-byte codepoint with
 * class 0 (matches nothing) so it falls to the final single-byte case.
 */

#define CLS_LETTER  X8R_CLS_LETTER
#define CLS_DIGIT   X8R_CLS_DIGIT
#define CLS_SPACE   X8R_CLS_SPACE
#define CLS_NEWLINE X8R_CLS_NEWLINE

typedef struct { uint32_t cls; uint32_t blen; } cp_info;

static inline cp_info cp_at(const uint8_t *b, size_t n, size_t p) {
    cp_info r = {0, 1};
    uint8_t c = b[p];
    uint32_t cp;
    if (c < 0x80) {
        cp = c;
    } else if ((c & 0xE0) == 0xC0 && p + 1 < n && (b[p+1] & 0xC0) == 0x80) {
        cp = ((uint32_t)(c & 0x1F) << 6) | (b[p+1] & 0x3F);
        if (cp < 0x80) return r;           /* overlong */
        r.blen = 2;
    } else if ((c & 0xF0) == 0xE0 && p + 2 < n && (b[p+1] & 0xC0) == 0x80 && (b[p+2] & 0xC0) == 0x80) {
        cp = ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(b[p+1] & 0x3F) << 6) | (b[p+2] & 0x3F);
        if (cp < 0x800) return r;          /* overlong */
        if (cp >= 0xD800 && cp <= 0xDFFF) return r;  /* surrogate */
        r.blen = 3;
    } else if ((c & 0xF8) == 0xF0 && p + 3 < n
               && (b[p+1] & 0xC0) == 0x80 && (b[p+2] & 0xC0) == 0x80 && (b[p+3] & 0xC0) == 0x80) {
        cp = ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(b[p+1] & 0x3F) << 12)
           | ((uint32_t)(b[p+2] & 0x3F) << 6) | (b[p+3] & 0x3F);
        if (cp < 0x10000 || cp > 0x10FFFF) return r;  /* overlong / out of range */
        r.blen = 4;
    } else {
        return r;  /* invalid: treat as single byte, no class */
    }
    r.cls = x8r_cp_class(cp);
    return r;
}

static inline int has_any(uint32_t cls, uint32_t mask) { return (cls & mask) != 0; }

/* rule 1: case-insensitive ASCII contractions. only matches ASCII letters. */
static size_t try_contraction(const uint8_t *b, size_t n, size_t p) {
    if (b[p] != '\'') return 0;
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

/* extend a letter run starting at r. uses AVX2 to bulk-scan ASCII
 * letters, falling back to cp_at for non-ASCII codepoints. */
static size_t extend_letter_run(const uint8_t *b, size_t n, size_t r) {
    for (;;) {
        r = x8r_scan_ascii_letter(b, n, r);
        if (r >= n) return r;
        /* either a non-letter ASCII byte or a non-ASCII byte. let
         * cp_at classify; if still a letter, advance and loop back
         * into SIMD. */
        cp_info c = cp_at(b, n, r);
        if (!(c.cls & CLS_LETTER)) return r;
        r += c.blen;
    }
}

/* rule 2: [^\r\n\p{L}\p{N}]? \p{L}+ */
static size_t try_letter_run(const uint8_t *b, size_t n, size_t p) {
    /* try with optional leading non-(letter|digit|newline) codepoint */
    cp_info lead = cp_at(b, n, p);
    size_t after_lead = p + lead.blen;
    int lead_ok = after_lead < n
                  && !has_any(lead.cls, CLS_LETTER | CLS_DIGIT | CLS_NEWLINE);
    if (lead_ok) {
        cp_info first = cp_at(b, n, after_lead);
        if (first.cls & CLS_LETTER) {
            size_t r = extend_letter_run(b, n, after_lead + first.blen);
            return r - p;
        }
    }
    /* without leading */
    cp_info first = cp_at(b, n, p);
    if (first.cls & CLS_LETTER) {
        size_t r = extend_letter_run(b, n, p + first.blen);
        return r - p;
    }
    return 0;
}

/* rule 3: \p{N}{1,3} */
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

/* rule 4:  ?[^\s\p{L}\p{N}]+[\r\n]*
 * note: the optional leading ' ' is a literal ASCII space, not \s. */
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
    while (q < n && (b[q] == '\r' || b[q] == '\n')) q++;
    return q - p;
}

/* rule 5: \s*[\r\n]+ (greedy-with-backtrack) */
static size_t try_space_then_newline(const uint8_t *b, size_t n, size_t p) {
    size_t q = p;
    for (;;) {
        q = x8r_scan_ascii_space(b, n, q);
        if (q >= n) break;
        cp_info c = cp_at(b, n, q);
        if (!(c.cls & CLS_SPACE)) break;
        q += c.blen;
    }
    /* shrink back to end on \r or \n */
    size_t r = q;
    while (r > p && b[r - 1] != '\r' && b[r - 1] != '\n') r--;
    if (r == p) return 0;
    return r - p;
}

/* rule 6: \s+(?!\S) */
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
    if (q == n) return q - p;  /* at EOF: full run */
    /* leave one codepoint of whitespace */
    size_t r = q;
    /* find the start of the last whitespace codepoint */
    size_t last = p;
    size_t walk = p;
    while (walk < q) {
        last = walk;
        cp_info c = cp_at(b, n, walk);
        walk += c.blen;
    }
    if (last == p) return 0;  /* only one whitespace cp, cannot leave one behind */
    return last - p;
    (void)r;
}

/* rule 7: \s+ */
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

size_t x8r_pretokenize_scalar(const uint8_t *buf, size_t len,
                              x8r_pretok_sink sink, void *user) {
    size_t p = 0;
    size_t count = 0;
    while (p < len) {
        size_t take = 0;

        if ((take = try_contraction(buf, len, p))) goto emit;
        if ((take = try_letter_run(buf, len, p))) goto emit;
        if ((take = try_digits(buf, len, p))) goto emit;
        if ((take = try_punct_run(buf, len, p))) goto emit;
        if ((take = try_space_then_newline(buf, len, p))) goto emit;
        if ((take = try_trailing_ws(buf, len, p))) goto emit;
        if ((take = try_any_ws(buf, len, p))) goto emit;

        /* nothing matched: emit one codepoint */
        {
            cp_info c = cp_at(buf, len, p);
            take = c.blen;
        }

    emit:
        sink(user, p, take);
        p += take;
        count++;
    }
    return count;
}
