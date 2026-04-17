#include "internal.h"

#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

/*
 * AVX2 accelerators for the two hottest inner loops in the
 * pre-tokenizer: ASCII letter runs (rule 2) and ASCII whitespace runs
 * (rules 5/6/7). both scan 32 bytes at a time and stop early on the
 * first non-ASCII byte so the scalar path can handle multi-byte
 * codepoints with full unicode classification.
 *
 * return value: the byte offset of the first position at or after `p`
 * that does NOT match the target class, OR is non-ASCII. callers must
 * treat the return position as "need to look at this byte scalar-
 * style" rather than "definitely outside the run".
 */

/* first non-ASCII-letter OR first non-ASCII byte in [p, n) */
size_t x8r_scan_ascii_letter(const uint8_t *b, size_t n, size_t p) {
    /* scalar head-align optional; the loop just strides in 32 */
    while (p + 32 <= n) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(b + p));

        /* detect any byte >= 0x80 (non-ASCII). mask has high bits set
         * wherever byte has bit 7 set. */
        int non_ascii_mask = _mm256_movemask_epi8(v);

        /* ASCII letter test: ((c | 0x20) - 'a') < 26, unsigned.
         * signed-only AVX2 comparisons, so shift the range into the
         * signed domain by subtracting 0x80 first. */
        __m256i lc = _mm256_or_si256(v, _mm256_set1_epi8(0x20));
        __m256i shifted = _mm256_sub_epi8(lc, _mm256_set1_epi8((char)(0x80 + 'a')));
        /* now "in range [a..z]" becomes shifted in [-0x80, -0x80 + 25]
         * i.e. shifted <= -0x80 + 25 = -103 (signed). use cmpgt. */
        __m256i in_range = _mm256_cmpgt_epi8(
            _mm256_set1_epi8((char)(-0x80 + 26)), shifted);
        int letter_mask = _mm256_movemask_epi8(in_range);

        /* want: first position that is not-a-letter OR non-ASCII */
        uint32_t bad = (uint32_t)(~letter_mask) | (uint32_t)non_ascii_mask;
        if (bad) {
            return p + (size_t)__builtin_ctz(bad);
        }
        p += 32;
    }
    /* tail: scalar handles it via the cp_at() loop in the caller */
    return p;
}

/* first non-ASCII-space OR first non-ASCII byte in [p, n).
 * ASCII whitespace per Unicode White_Space: \t \n \v \f \r ' ' and also
 * 0x1C..0x1F? no — those aren't in White_Space. just the six classic. */
size_t x8r_scan_ascii_space(const uint8_t *b, size_t n, size_t p) {
    const __m256i tab = _mm256_set1_epi8('\t');
    const __m256i lf  = _mm256_set1_epi8('\n');
    const __m256i vt  = _mm256_set1_epi8(0x0B);
    const __m256i ff  = _mm256_set1_epi8(0x0C);
    const __m256i cr  = _mm256_set1_epi8('\r');
    const __m256i sp  = _mm256_set1_epi8(' ');

    while (p + 32 <= n) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(b + p));
        int non_ascii_mask = _mm256_movemask_epi8(v);

        __m256i s =
            _mm256_or_si256(
                _mm256_or_si256(_mm256_cmpeq_epi8(v, tab), _mm256_cmpeq_epi8(v, lf)),
                _mm256_or_si256(
                    _mm256_or_si256(_mm256_cmpeq_epi8(v, vt), _mm256_cmpeq_epi8(v, ff)),
                    _mm256_or_si256(_mm256_cmpeq_epi8(v, cr), _mm256_cmpeq_epi8(v, sp))));
        int space_mask = _mm256_movemask_epi8(s);

        uint32_t bad = (uint32_t)(~space_mask) | (uint32_t)non_ascii_mask;
        if (bad) {
            return p + (size_t)__builtin_ctz(bad);
        }
        p += 32;
    }
    return p;
}
