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
 * two paths: a linear scan for short pre-tokens (<=32 parts) and a
 * binary min-heap for longer ones (O(n log n) vs O(n^2)).
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

/* binary min-heap of (rank, index) pairs for the merge loop */
typedef struct {
    uint32_t rank;
    int32_t idx;
} heap_entry;

static void heap_sift_down(heap_entry *h, size_t n, size_t i) {
    for (;;) {
        size_t smallest = i;
        size_t l = 2 * i + 1;
        size_t r = 2 * i + 2;
        if (l < n && h[l].rank < h[smallest].rank) smallest = l;
        if (r < n && h[r].rank < h[smallest].rank) smallest = r;
        if (smallest == i) break;
        heap_entry t = h[i]; h[i] = h[smallest]; h[smallest] = t;
        i = smallest;
    }
}

static void heap_push(heap_entry *h, size_t *n, uint32_t rank, int32_t idx) {
    size_t i = (*n)++;
    h[i].rank = rank;
    h[i].idx = idx;
    while (i > 0) {
        size_t p = (i - 1) / 2;
        if (h[p].rank <= h[i].rank) break;
        heap_entry t = h[p]; h[p] = h[i]; h[i] = t;
        i = p;
    }
}

static void heap_pop(heap_entry *h, size_t *n) {
    if (*n == 0) return;
    h[0] = h[--(*n)];
    if (*n > 0) heap_sift_down(h, *n, 0);
}

size_t x8r_bpe_encode(const x8r_vocab *v,
                      const uint8_t *bytes, size_t len,
                      uint32_t **ranks_out, size_t *ranks_cap, size_t *ranks_len) {
    if (len == 0) return 0;

    /* count-only mode: ranks_out == NULL means caller only wants the token
     * count. skips all ranks-buffer growth and the final per-part rank
     * lookup (we already know part count = token count). */
    const int count_only = (ranks_out == NULL);

    /* fast path: whole pre-token is a single vocab token. common for short runs. */
    uint32_t whole = x8r_vocab_lookup(v, bytes, len);
    if (whole != UINT32_MAX) {
        if (count_only) return 1;
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

    /* threshold split: linear scan for short pre-tokens (common in
     * generated corpus), binary min-heap for longer ones */
    if (len <= 32) {

        /* --- linear scan (kept exactly as-is for len <= 32) --- */
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

    } else {

        /* --- binary min-heap path for long pre-tokens (len > 32) --- */
        heap_entry hbuf[512];
        heap_entry *hp;
        int hp_heap = 0;
        if (len <= 512) {
            hp = hbuf;
        } else {
            hp = (heap_entry *)malloc(sizeof(heap_entry) * (len * 2));
            if (!hp) { if (heap) free(parts); return 0; }
            hp_heap = 1;
        }

        size_t hp_n = 0;
        for (size_t i = 0; i < len; ++i) {
            if (parts[i].next >= 0) {
                heap_push(hp, &hp_n, parts[i].rank, (int32_t)i);
            }
        }

        for (;;) {
            int32_t best = -1;
            while (hp_n > 0) {
                uint32_t r = hp[0].rank;
                int32_t idx = hp[0].idx;
                heap_pop(hp, &hp_n);
                if (parts[idx].next >= 0 && parts[idx].rank == r) {
                    best = idx;
                    break;
                }
            }
            if (best < 0) break;

            int32_t right = parts[best].next;
            parts[best].len += parts[right].len;
            parts[best].next = parts[right].next;
            if (parts[best].next >= 0) parts[parts[best].next].prev = best;
            parts[right].next = -1;      /* invalidate stale heap entries */

            if (parts[best].prev >= 0) {
                int32_t pv = parts[best].prev;
                parts[pv].rank = pair_rank(v, bytes, &parts[pv], &parts[best]);
                heap_push(hp, &hp_n, parts[pv].rank, pv);
            }
            if (parts[best].next >= 0) {
                parts[best].rank = pair_rank(v, bytes, &parts[best], &parts[parts[best].next]);
                heap_push(hp, &hp_n, parts[best].rank, best);
            } else {
                parts[best].rank = UINT32_MAX;
            }
        }

        if (hp_heap) free(hp);
    }

    /* emit remaining parts as tokens */
    size_t emitted = 0;
    int32_t cur = 0;
    if (count_only) {
        /* count-only: skip per-part vocab_lookup (we only need the count) */
        while (cur >= 0) { emitted++; cur = parts[cur].next; }
    } else {
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
    }

    if (heap) free(parts);
    return emitted;
}