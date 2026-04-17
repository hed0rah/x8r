#include "internal.h"

#include <stdlib.h>
#include <string.h>

/*
 * classic byte-pair encoding applied to a single pre-token.
 *
 * representation during merging: a doubly-linked list of "parts", each
 * pointing at a slice of the original bytes. we repeatedly find the
 * adjacent pair with the lowest rank (highest priority merge) and
 * combine it. when no merge is possible the remaining parts are the
 * final tokens.
 *
 * this is the straightforward O(n^2) implementation. a priority queue
 * would drop it to O(n log n), worth it once profiling shows it.
 */

typedef struct {
    uint32_t start;   /* byte offset within pre-token */
    uint32_t len;     /* length of this part in bytes */
    int32_t prev;
    int32_t next;
    uint32_t rank;    /* rank of the merge of (this, next), UINT32_MAX if none */
} bpe_part;

static uint32_t pair_rank(const x8r_vocab *v, const uint8_t *bytes,
                          const bpe_part *a, const bpe_part *b) {
    uint32_t total = a->len + b->len;
    if (total > 0xFFFF) return UINT32_MAX;
    /* bytes are contiguous in the source by construction */
    return x8r_vocab_lookup(v, bytes + a->start, total);
}

static size_t ensure_cap(uint32_t **ranks, size_t *cap, size_t need) {
    if (need <= *cap) return 0;
    size_t ncap = *cap ? *cap * 2 : 64;
    while (ncap < need) ncap *= 2;
    uint32_t *nr = (uint32_t *)realloc(*ranks, ncap * sizeof(uint32_t));
    if (!nr) return (size_t)-1;
    *ranks = nr;
    *cap = ncap;
    return 0;
}

size_t x8r_bpe_encode(const x8r_vocab *v,
                      const uint8_t *bytes, size_t len,
                      uint32_t **ranks_out, size_t *ranks_cap, size_t *ranks_len) {
    if (len == 0) return 0;

    /* fast path: whole pre-token is a single vocab token. common for short runs. */
    uint32_t whole = x8r_vocab_lookup(v, bytes, len);
    if (whole != UINT32_MAX) {
        if (ensure_cap(ranks_out, ranks_cap, *ranks_len + 1) == (size_t)-1) return 0;
        (*ranks_out)[(*ranks_len)++] = whole;
        return 1;
    }

    /* build a byte-per-part doubly-linked list on the stack / small heap */
    bpe_part stackbuf[512];
    bpe_part *parts;
    int heap = 0;
    if (len + 1 <= sizeof(stackbuf)/sizeof(stackbuf[0])) {
        parts = stackbuf;
    } else {
        parts = (bpe_part *)malloc(sizeof(bpe_part) * (len + 1));
        if (!parts) return 0;
        heap = 1;
    }

    for (size_t i = 0; i < len; ++i) {
        parts[i].start = (uint32_t)i;
        parts[i].len = 1;
        parts[i].prev = (i == 0) ? -1 : (int32_t)(i - 1);
        parts[i].next = (i + 1 == len) ? -1 : (int32_t)(i + 1);
        parts[i].rank = UINT32_MAX;
    }

    /* seed pair ranks */
    for (size_t i = 0; i < len; ++i) {
        if (parts[i].next >= 0) {
            parts[i].rank = pair_rank(v, bytes, &parts[i], &parts[parts[i].next]);
        }
    }

    /* merge until no pair is in the vocab */
    for (;;) {
        int32_t best = -1;
        uint32_t best_rank = UINT32_MAX;
        int32_t cur = 0;
        while (cur >= 0) {
            if (parts[cur].next >= 0 && parts[cur].rank < best_rank) {
                best_rank = parts[cur].rank;
                best = cur;
            }
            cur = parts[cur].next;
        }
        if (best < 0) break;

        /* merge best and best->next */
        int32_t right = parts[best].next;
        parts[best].len += parts[right].len;
        parts[best].next = parts[right].next;
        if (parts[best].next >= 0) parts[parts[best].next].prev = best;

        /* recompute ranks for (prev, best) and (best, next) */
        if (parts[best].prev >= 0) {
            int32_t pv = parts[best].prev;
            parts[pv].rank = pair_rank(v, bytes, &parts[pv], &parts[best]);
        }
        if (parts[best].next >= 0) {
            parts[best].rank = pair_rank(v, bytes, &parts[best], &parts[parts[best].next]);
        } else {
            parts[best].rank = UINT32_MAX;
        }
    }

    /* emit remaining parts as tokens */
    size_t emitted = 0;
    int32_t cur = 0;
    while (cur >= 0) {
        uint32_t rank = x8r_vocab_lookup(v, bytes + parts[cur].start, parts[cur].len);
        if (rank == UINT32_MAX) {
            /* unknown single byte: this indicates a vocab problem.
             * fallback: emit per-byte with 0 rank to keep the stream
             * flowing. golden tests will fail, which is what we want. */
            rank = 0;
        }
        if (ensure_cap(ranks_out, ranks_cap, *ranks_len + 1) == (size_t)-1) {
            if (heap) free(parts);
            return emitted;
        }
        (*ranks_out)[(*ranks_len)++] = rank;
        emitted++;
        cur = parts[cur].next;
    }

    if (heap) free(parts);
    return emitted;
}
