#define _POSIX_C_SOURCE 200809L
#include "x8r.h"
#include "internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void usage(FILE *f) {
    fputs(
        "usage: x8r [options] <file>\n"
        "       x8r [options] -   (read stdin)\n"
        "\n"
        "options:\n"
        "  --budget N           chunk budget in tokens (default: count-only mode)\n"
        "  --vocab PATH         path to vocab binary (default: $X8R_VOCAB or ./vocab/cl100k.bin)\n"
        "  --model NAME         cl100k | o200k | auto (default: auto, trust blob)\n"
        "  --boundary MODE      none | line | auto (default: line)\n"
        "  --tolerance F        rewind window as fraction of budget (default 0.10)\n"
        "  --json               emit json output\n"
        "  --count              just print the total token count\n"
        "  --encode             print token ids, one per line (or json array)\n"
        "  --decode             read ids from stdin/file (one per line), write raw bytes\n"
        "  -h, --help\n",
        f);
}

static int read_stdin(uint8_t **buf_out, size_t *len_out) {
    size_t cap = 1 << 16, n = 0;
    uint8_t *buf = malloc(cap);
    if (!buf) return -1;
    for (;;) {
        if (n == cap) {
            cap *= 2;
            uint8_t *nb = realloc(buf, cap);
            if (!nb) { free(buf); return -1; }
            buf = nb;
        }
        ssize_t r = read(0, buf + n, cap - n);
        if (r < 0) { free(buf); return -1; }
        if (r == 0) break;
        n += (size_t)r;
    }
    *buf_out = buf;
    *len_out = n;
    return 0;
}

static const char *cut_name(x8r_cut_kind k) {
    switch (k) {
        case X8R_CUT_EOF: return "eof";
        case X8R_CUT_LINE: return "line";
        case X8R_CUT_BLANK: return "blank";
        case X8R_CUT_BLOCK: return "block";
        case X8R_CUT_HARD: default: return "hard";
    }
}

int main(int argc, char **argv) {
    const char *path = NULL;
    const char *vocab_path = getenv("X8R_VOCAB");
    if (!vocab_path) vocab_path = "./vocab/cl100k.bin";
    size_t budget = 0;
    int want_json = 0;
    int count_only = 0;
    int encode_only = 0;
    int decode_only = 0;
    x8r_boundary_mode boundary = X8R_BOUNDARY_LINE;
    double tolerance = 0.10;
    x8r_vocab_id model = X8R_VOCAB_AUTO;

    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(stdout); return 0; }
        else if (!strcmp(a, "--budget") && i + 1 < argc) { budget = strtoull(argv[++i], NULL, 10); }
        else if (!strcmp(a, "--vocab")  && i + 1 < argc) { vocab_path = argv[++i]; }
        else if (!strcmp(a, "--model")  && i + 1 < argc) {
            const char *m = argv[++i];
            if (!strcmp(m, "cl100k")) model = X8R_VOCAB_CL100K;
            else if (!strcmp(m, "o200k")) model = X8R_VOCAB_O200K;
            else if (!strcmp(m, "auto")) model = X8R_VOCAB_AUTO;
            else { fprintf(stderr, "unknown model: %s\n", m); return 2; }
        }
        else if (!strcmp(a, "--boundary") && i + 1 < argc) {
            const char *m = argv[++i];
            if (!strcmp(m, "none")) boundary = X8R_BOUNDARY_NONE;
            else if (!strcmp(m, "line")) boundary = X8R_BOUNDARY_LINE;
            else if (!strcmp(m, "auto")) boundary = X8R_BOUNDARY_AUTO;
            else { fprintf(stderr, "unknown boundary mode: %s\n", m); return 2; }
        }
        else if (!strcmp(a, "--tolerance") && i + 1 < argc) { tolerance = strtod(argv[++i], NULL); }
        else if (!strcmp(a, "--json")) { want_json = 1; }
        else if (!strcmp(a, "--count")) { count_only = 1; }
        else if (!strcmp(a, "--encode")) { encode_only = 1; }
        else if (!strcmp(a, "--decode")) { decode_only = 1; }
        else if (a[0] == '-' && a[1] != '\0' && a[1] != '-') { fprintf(stderr, "unknown flag: %s\n", a); return 2; }
        else { path = a; }
    }
    if (!path) { usage(stderr); return 2; }

    const uint8_t *buf = NULL;
    size_t len = 0;
    void *handle = NULL;
    uint8_t *stdin_buf = NULL;

    if (!strcmp(path, "-")) {
        if (read_stdin(&stdin_buf, &len) != 0) { fprintf(stderr, "stdin read failed\n"); return 1; }
        buf = stdin_buf;
    } else {
        x8r_status st = x8r_mmap_ro(path, &buf, &len, &handle);
        if (st != X8R_OK) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    }

    x8r_ctx *ctx = NULL;
    x8r_status st = x8r_ctx_open(vocab_path, model, &ctx);
    if (st != X8R_OK) {
        fprintf(stderr, "cannot load vocab %s (status %d)\n", vocab_path, st);
        return 1;
    }

    int rc = 0;
    if (decode_only) {
        /* parse decimal ids from buf, one per whitespace-separated token. */
        size_t cap = 1024, n = 0;
        uint32_t *ids = malloc(cap * sizeof(uint32_t));
        if (!ids) { fprintf(stderr, "oom\n"); rc = 1; goto cleanup; }
        size_t i = 0;
        while (i < len) {
            while (i < len && (buf[i] == ' ' || buf[i] == '\t' || buf[i] == '\n'
                               || buf[i] == '\r' || buf[i] == ',' || buf[i] == '['
                               || buf[i] == ']')) i++;
            if (i >= len) break;
            uint64_t val = 0;
            int saw_digit = 0;
            while (i < len && buf[i] >= '0' && buf[i] <= '9') {
                val = val * 10 + (buf[i] - '0');
                if (val > 0xFFFFFFFFu) { fprintf(stderr, "id overflow\n"); rc = 1; free(ids); goto cleanup; }
                saw_digit = 1;
                i++;
            }
            if (!saw_digit) { fprintf(stderr, "non-numeric in id list at byte %zu\n", i); rc = 1; free(ids); goto cleanup; }
            if (n == cap) {
                cap *= 2;
                uint32_t *nb = realloc(ids, cap * sizeof(uint32_t));
                if (!nb) { fprintf(stderr, "oom\n"); rc = 1; free(ids); goto cleanup; }
                ids = nb;
            }
            ids[n++] = (uint32_t)val;
        }
        uint8_t *out_bytes = NULL;
        size_t out_n = 0;
        st = x8r_decode_bytes(ctx, ids, n, &out_bytes, &out_n);
        if (st != X8R_OK) {
            fprintf(stderr, "decode failed: %d (status %d)\n",
                    st == X8R_E_VOCAB ? 1 : 2, st);
            rc = 1;
        } else {
            if (out_n > 0) fwrite(out_bytes, 1, out_n, stdout);
            x8r_bytes_free(out_bytes);
        }
        free(ids);
    } else if (encode_only) {
        uint32_t *ids = NULL;
        size_t n = 0;
        st = x8r_encode_ordinary(ctx, buf, len, &ids, &n);
        if (st != X8R_OK) { fprintf(stderr, "encode failed: %d\n", st); rc = 1; }
        else if (want_json) {
            fputc('[', stdout);
            for (size_t i = 0; i < n; ++i) printf("%s%u", i ? "," : "", ids[i]);
            fputs("]\n", stdout);
        } else {
            for (size_t i = 0; i < n; ++i) printf("%u\n", ids[i]);
        }
        x8r_ids_free(ids);
    } else if (count_only || budget == 0) {
        size_t n = x8r_count_tokens(ctx, buf, len);
        if (want_json) printf("{\"tokens\":%zu,\"bytes\":%zu}\n", n, len);
        else printf("%zu\n", n);
    } else {
        x8r_opts opts = { .budget = budget, .vocab = model,
                          .boundary = boundary, .tolerance = tolerance };
        x8r_chunk *chunks = NULL;
        size_t n = 0;
        st = x8r_chunk_buf(ctx, buf, len, &opts, &chunks, &n);
        if (st != X8R_OK) { fprintf(stderr, "chunk failed: %d\n", st); rc = 1; }
        else if (want_json) {
            fputs("[", stdout);
            for (size_t i = 0; i < n; ++i) {
                printf("%s{\"start\":%zu,\"end\":%zu,\"tokens\":%zu,\"cut\":\"%s\"}",
                       i ? "," : "",
                       chunks[i].start, chunks[i].end, chunks[i].token_count,
                       cut_name(chunks[i].cut));
            }
            fputs("]\n", stdout);
        } else {
            for (size_t i = 0; i < n; ++i) {
                printf("%zu\t%zu\t%zu\t%s\n",
                       chunks[i].start, chunks[i].end, chunks[i].token_count,
                       cut_name(chunks[i].cut));
            }
        }
        x8r_chunks_free(chunks);
    }

cleanup:
    x8r_ctx_close(ctx);
    if (handle) x8r_munmap(handle, len);
    free(stdin_buf);
    return rc;
}
