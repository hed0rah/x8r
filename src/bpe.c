#include "internal.h"

#include <stdlib.h>
#include <string.h>

/*
 * byte-pair encoding applied to a single pre-token.
 *
 * representation: a doubly-linked list of "parts", each pointing at a
 * slice of the original bytes, plus a binary min-heap of (rank, idx)
 * entries used to find the best merge in O(log n) per step.
 *
 * staleness handling: refresh-and-repush. on pop:
 *   1. parts[idx].next < 0 means the part was consumed by an earlier
 *      merge: drop the entry (dead-flag).
 *   2. let cur = parts[idx].rank. cur != popped rank means the pair at
 *      idx changed since this entry was pushed. push (cur, idx) back if
 *      cur is valid, otherwise drop. continue.
 *   3. otherwise the entry is current: do the merge.
 *
 * the heap is structurally drain-immune: stale-rank pops produce a
 * refresh push (1:1, no net change), dead-flag pops are bounded 1:1
 * with merges. the heap empties only when no valid pair remains.
 *
 * see notes/bpe-heap-survey.md for the comparison vs huggingface
 * tokenizers and tiktoken.
 */

typedef struct {
    uint32_t start;     /* byte offset within pre-token */
    uint32_t len;       /* length of this part in bytes */
    int32_t prev;
    int32_t next;       /* -1 means consumed (dead-flag) or end-of-list */
    uint32_t rank;      /* rank of (this, next), UINT32_MAX if no merge */
} bpe_part;

typedef struct {
    uint32_t rank;
    int32_t idx;        /* index into parts[] of the left side of the pair */
} bpe_heap_entry;

/* stack heap: covers most pre-tokens (initial seed + 2/merge ~= 3*len pushes
 * worst case, but heap size at any moment is much smaller). 512 entries =
 * 4 KB stack, comfortable. longer pre-tokens fall through to malloc. */
#define BPE_HEAP_STACK_CAP 512

typedef struct {
    bpe_heap_entry *data;
    size_t size;
    size_t cap;
    bpe_heap_entry stackbuf[BPE_HEAP_STACK_CAP];
    int on_heap;
} bpe_heap;

static int heap_init(bpe_heap *h, size_t hint) {
    h->size = 0;
    if (hint <= BPE_HEAP_STACK_CAP) {
        h->data = h->stackbuf;
        h->cap = BPE_HEAP_STACK_CAP;
        h->on_heap = 0;
        return 0;
    }
    size_t cap = 64;
    while (cap < hint) cap *= 2;
    h->data = (bpe_heap_entry *)malloc(cap * sizeof(bpe_heap_entry));
    if (!h->data) { h->cap = 0; return -1; }
    h->cap = cap;
    h->on_heap = 1;
    return 0;
}

static void heap_free(bpe_heap *h) {
    if (h->on_heap && h->data) free(h->data);
}

static int heap_grow(bpe_heap *h) {
    size_t ncap = h->cap ? h->cap * 2 : 64;
    bpe_heap_entry *nd;
    if (h->on_heap) {
        nd = (bpe_heap_entry *)realloc(h->data, ncap * sizeof(bpe_heap_entry));
        if (!nd) return -1;
    } else {
        nd = (bpe_heap_entry *)malloc(ncap * sizeof(bpe_heap_entry));
        if (!nd) return -1;
        memcpy(nd, h->data, h->size * sizeof(bpe_heap_entry));
        h->on_heap = 1;
    }
    h->data = nd;
    h->cap = ncap;
    return 0;
}

/* a < b iff (a.rank < b.rank) || (a.rank == b.rank && a.idx < b.idx).
 * idx tie-break preserves the leftmost-wins property of the prior
 * linear scan, which is required for bit-exact tiktoken parity. */
static inline int entry_lt(const bpe_heap_entry *a, const bpe_heap_entry *b) {
    if (a->rank != b->rank) return a->rank < b->rank;
    return a->idx < b->idx;
}

static int heap_push(bpe_heap *h, uint32_t rank, int32_t idx) {
    if (h->size + 1 > h->cap) {
        if (heap_grow(h) < 0) return -1;
    }
    size_t i = h->size++;
    h->data[i].rank = rank;
    h->data[i].idx = idx;
    while (i > 0) {
        size_t p = (i - 1) / 2;
        if (!entry_lt(&h->data[i], &h->data[p])) break;
        bpe_heap_entry t = h->data[i];
        h->data[i] = h->data[p];
        h->data[p] = t;
        i = p;
    }
    return 0;
}

static int heap_pop(bpe_heap *h, bpe_heap_entry *out) {
    if (h->size == 0) return -1;
    *out = h->data[0];
    h->size--;
    if (h->size > 0) {
        h->data[0] = h->data[h->size];
        size_t i = 0;
        for (;;) {
            size_t l = 2*i + 1, r = 2*i + 2, s = i;
            if (l < h->size && entry_lt(&h->data[l], &h->data[s])) s = l;
            if (r < h->size && entry_lt(&h->data[r], &h->data[s])) s = r;
            if (s == i) break;
            bpe_heap_entry t = h->data[i];
            h->data[i] = h->data[s];
            h->data[s] = t;
            i = s;
        }
    }
    return 0;
}

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

    /* count-only mode: ranks_out == NULL means caller only wants the token
     * count. skips ranks-buffer growth and the final per-part rank lookup. */
    const int count_only = (ranks_out == NULL);

    /* fast path: whole pre-token is a single vocab token. common for short runs. */
    uint32_t whole = x8r_vocab_lookup(v, bytes, len);
    if (whole != UINT32_MAX) {
        if (count_only) return 1;
        if (ensure_cap(ranks_out, ranks_cap, *ranks_len + 1) == (size_t)-1) return 0;
        (*ranks_out)[(*ranks_len)++] = whole;
        return 1;
    }

    /* parts buffer: stack for short pre-tokens, malloc for long ones */
    bpe_part stackparts[512];
    bpe_part *parts;
    int parts_on_heap = 0;
    if (len + 1 <= sizeof(stackparts)/sizeof(stackparts[0])) {
        parts = stackparts;
    } else {
        parts = (bpe_part *)malloc(sizeof(bpe_part) * (len + 1));
        if (!parts) return 0;
        parts_on_heap = 1;
    }

    for (size_t i = 0; i < len; ++i) {
        parts[i].start = (uint32_t)i;
        parts[i].len = 1;
        parts[i].prev = (i == 0) ? -1 : (int32_t)(i - 1);
        parts[i].next = (i + 1 == len) ? -1 : (int32_t)(i + 1);
        parts[i].rank = UINT32_MAX;
    }

    /* heap: hint at 4*len, generous since we may push 1 init + 2/merge.
     * actual heap size at any moment is much smaller (refresh-and-repush
     * doesn't grow the heap), but giving a high hint avoids growth churn. */
    bpe_heap heap;
    if (heap_init(&heap, 4 * (len + 1)) < 0) {
        if (parts_on_heap) free(parts);
        return 0;
    }

    /* seed pair ranks and heap */
    for (size_t i = 0; i < len; ++i) {
        if (parts[i].next < 0) continue;
        uint32_t r = pair_rank(v, bytes, &parts[i], &parts[parts[i].next]);
        parts[i].rank = r;
        if (r != UINT32_MAX) {
            if (heap_push(&heap, r, (int32_t)i) < 0) {
                heap_free(&heap);
                if (parts_on_heap) free(parts);
                return 0;
            }
        }
    }

    /* merge loop */
    bpe_heap_entry top;
    while (heap_pop(&heap, &top) == 0) {
        int32_t i = top.idx;

        /* dead-flag: part was consumed by an earlier merge */
        if (parts[i].next < 0) continue;

        uint32_t cur = parts[i].rank;
        if (cur != top.rank) {
            /* stale: refresh from authoritative parts[i].rank.
             * if still valid, repush. if no longer valid (UINT32_MAX),
             * drop -- bounded 1:1 with the merge that invalidated it. */
            if (cur != UINT32_MAX) {
                if (heap_push(&heap, cur, i) < 0) {
                    heap_free(&heap);
                    if (parts_on_heap) free(parts);
                    return 0;
                }
            }
            continue;
        }

        /* current: merge i with i.next */
        int32_t r = parts[i].next;
        parts[i].len += parts[r].len;
        parts[i].next = parts[r].next;
        if (parts[i].next >= 0) parts[parts[i].next].prev = i;
        parts[r].next = -1;  /* dead-flag the consumed part */

        /* recompute and push the two affected pair ranks */
        if (parts[i].prev >= 0) {
            int32_t pv = parts[i].prev;
            uint32_t pr = pair_rank(v, bytes, &parts[pv], &parts[i]);
            parts[pv].rank = pr;
            if (pr != UINT32_MAX) {
                if (heap_push(&heap, pr, pv) < 0) {
                    heap_free(&heap);
                    if (parts_on_heap) free(parts);
                    return 0;
                }
            }
        }
        if (parts[i].next >= 0) {
            uint32_t nr = pair_rank(v, bytes, &parts[i], &parts[parts[i].next]);
            parts[i].rank = nr;
            if (nr != UINT32_MAX) {
                if (heap_push(&heap, nr, i) < 0) {
                    heap_free(&heap);
                    if (parts_on_heap) free(parts);
                    return 0;
                }
            }
        } else {
            parts[i].rank = UINT32_MAX;
        }
    }

    heap_free(&heap);

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
                if (parts_on_heap) free(parts);
                return emitted;
            }
            (*ranks_out)[(*ranks_len)++] = rank;
            emitted++;
            cur = parts[cur].next;
        }
    }

    if (parts_on_heap) free(parts);
    return emitted;
}
