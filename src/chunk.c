#define _POSIX_C_SOURCE 200809L
#include "internal.h"

#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

/*
 * milestone 1 strategy:
 *   - run pre-tokenizer over the buffer, collecting (byte_end, tok_count)
 *     at every pre-token boundary
 *   - also record line-end positions (every '\n') and their cumulative
 *     token index
 *   - for each chunk: advance until prefix tokens == budget; within the
 *     tolerance window, rewind to the nearest line boundary
 *
 * this is intentionally simple. later we'll add block/blank-line/syntax
 * boundaries with better scoring.
 */

typedef struct {
    const uint8_t *buf;
    size_t len;
    /* per pre-token */
    size_t *pt_end;      /* byte offset right after pre-token */
    size_t *pt_tok_cum;  /* cumulative tokens after this pre-token */
    size_t pt_cap;
    size_t pt_n;
    /* ranks scratch reused across pre-tokens */
    uint32_t *ranks;
    size_t ranks_cap;
    size_t ranks_len;
    const x8r_vocab *vocab;
} build_state;

typedef struct {
    const uint8_t *buf;
    build_state *st;
} sink_ctx;

static int push_pt(build_state *s, size_t end, size_t tok_cum) {
    if (s->pt_n == s->pt_cap) {
        size_t ncap = s->pt_cap ? s->pt_cap * 2 : 1024;
        size_t *ne = realloc(s->pt_end, ncap * sizeof(size_t));
        size_t *nt = realloc(s->pt_tok_cum, ncap * sizeof(size_t));
        if (!ne || !nt) { free(ne); free(nt); return -1; }
        s->pt_end = ne;
        s->pt_tok_cum = nt;
        s->pt_cap = ncap;
    }
    s->pt_end[s->pt_n] = end;
    s->pt_tok_cum[s->pt_n] = tok_cum;
    s->pt_n++;
    return 0;
}

static void pretok_sink_build(void *user, size_t start, size_t plen) {
    sink_ctx *c = (sink_ctx *)user;
    build_state *s = c->st;
    size_t before = s->ranks_len;
    x8r_bpe_encode(s->vocab, c->buf + start, plen,
                   &s->ranks, &s->ranks_cap, &s->ranks_len);
    size_t cum = s->ranks_len;
    (void)before;
    push_pt(s, start + plen, cum);
}

size_t x8r_count_tokens(x8r_ctx *ctx, const uint8_t *buf, size_t len) {
    build_state s = {0};
    s.buf = buf; s.len = len; s.vocab = &ctx->vocab;
    sink_ctx c = { buf, &s };
    x8r_pretokenize_scalar(buf, len, pretok_sink_build, &c);
    size_t n = s.ranks_len;
    free(s.pt_end); free(s.pt_tok_cum); free(s.ranks);
    return n;
}

/* find largest i with pt_tok_cum[i] <= target; returns -1 if none */
static ssize_t upper_bound(const size_t *a, size_t n, size_t target) {
    ssize_t lo = 0, hi = (ssize_t)n - 1, ans = -1;
    while (lo <= hi) {
        ssize_t mid = (lo + hi) / 2;
        if (a[mid] <= target) { ans = mid; lo = mid + 1; }
        else hi = mid - 1;
    }
    return ans;
}

x8r_status x8r_chunk_buf(x8r_ctx *ctx,
                         const uint8_t *buf, size_t len,
                         const x8r_opts *opts,
                         x8r_chunk **out_chunks, size_t *out_n) {
    if (!ctx || !opts || !out_chunks || !out_n) return X8R_E_ARG;
    if (opts->budget == 0) return X8R_E_ARG;

    build_state s = {0};
    s.buf = buf; s.len = len; s.vocab = &ctx->vocab;
    sink_ctx c = { buf, &s };
    x8r_pretokenize_scalar(buf, len, pretok_sink_build, &c);

    size_t total_tokens = s.ranks_len;

    double tol = opts->tolerance > 0 ? opts->tolerance : 0.10;
    size_t cap = 16;
    size_t n = 0;
    x8r_chunk *chunks = malloc(cap * sizeof(x8r_chunk));
    if (!chunks) {
        free(s.pt_end); free(s.pt_tok_cum); free(s.ranks);
        return X8R_E_NOMEM;
    }

    size_t cur_byte = 0;
    size_t cur_tok_base = 0;  /* cumulative tokens consumed so far */
    size_t start_pt = 0;      /* first pre-token index in remaining region */

    while (cur_tok_base < total_tokens) {
        size_t target = cur_tok_base + opts->budget;

        ssize_t pt_idx;
        x8r_cut_kind kind;
        if (target >= total_tokens) {
            pt_idx = (ssize_t)s.pt_n - 1;
            kind = X8R_CUT_EOF;
        } else {
            pt_idx = upper_bound(s.pt_tok_cum, s.pt_n, target);
            if (pt_idx < (ssize_t)start_pt) {
                /* budget too small to fit even one pre-token: take one */
                pt_idx = (ssize_t)start_pt;
            }
            kind = X8R_CUT_HARD;

            if (opts->boundary == X8R_BOUNDARY_LINE ||
                opts->boundary == X8R_BOUNDARY_SYNTAX_LIGHT ||
                opts->boundary == X8R_BOUNDARY_AUTO) {
                size_t min_tokens = cur_tok_base + (size_t)(opts->budget * (1.0 - tol));
                /* rewind pt_idx to a pre-token whose bytes end with '\n' */
                ssize_t r = pt_idx;
                while (r >= (ssize_t)start_pt && s.pt_tok_cum[r] >= min_tokens) {
                    size_t byte_end = s.pt_end[r];
                    if (byte_end > 0 && buf[byte_end - 1] == '\n') {
                        pt_idx = r;
                        kind = X8R_CUT_LINE;
                        break;
                    }
                    r--;
                }
            }
        }

        size_t byte_end = s.pt_end[pt_idx];
        size_t tok_cum = s.pt_tok_cum[pt_idx];

        if (n == cap) {
            cap *= 2;
            x8r_chunk *nc = realloc(chunks, cap * sizeof(x8r_chunk));
            if (!nc) {
                free(chunks);
                free(s.pt_end); free(s.pt_tok_cum); free(s.ranks);
                return X8R_E_NOMEM;
            }
            chunks = nc;
        }

        chunks[n].start = cur_byte;
        chunks[n].end = byte_end;
        chunks[n].token_count = tok_cum - cur_tok_base;
        chunks[n].cut = kind;
        n++;

        cur_byte = byte_end;
        cur_tok_base = tok_cum;
        start_pt = (size_t)pt_idx + 1;
    }

    free(s.pt_end); free(s.pt_tok_cum); free(s.ranks);
    *out_chunks = chunks;
    *out_n = n;
    return X8R_OK;
}

void x8r_chunks_free(x8r_chunk *chunks) { free(chunks); }

x8r_status x8r_ctx_open(const char *vocab_path, x8r_vocab_id vocab, x8r_ctx **out) {
    if (!vocab_path || !out) return X8R_E_ARG;
    x8r_ctx *c = calloc(1, sizeof(*c));
    if (!c) return X8R_E_NOMEM;
    x8r_status st = x8r_vocab_load(vocab_path, &c->vocab);
    if (st != X8R_OK) { free(c); return st; }
    c->id = vocab;
    *out = c;
    return X8R_OK;
}

void x8r_ctx_close(x8r_ctx *ctx) {
    if (!ctx) return;
    x8r_vocab_close(&ctx->vocab);
    free(ctx);
}

const char *x8r_version(void) { return "x8r 0.1.0"; }
