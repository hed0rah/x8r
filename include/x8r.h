#ifndef X8R_H
#define X8R_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define X8R_VERSION_MAJOR 0
#define X8R_VERSION_MINOR 1
#define X8R_VERSION_PATCH 0

typedef enum {
    X8R_OK = 0,
    X8R_E_IO = -1,
    X8R_E_VOCAB = -2,
    X8R_E_ARG = -3,
    X8R_E_NOMEM = -4,
    X8R_E_INTERNAL = -5,
} x8r_status;

typedef enum {
    X8R_VOCAB_CL100K = 0,
    X8R_VOCAB_O200K  = 1,
    X8R_VOCAB_AUTO   = 0xFFFF,  /* trust the blob's vocab_id */
} x8r_vocab_id;

typedef enum {
    X8R_BOUNDARY_NONE = 0,
    X8R_BOUNDARY_LINE = 1,
    X8R_BOUNDARY_SYNTAX_LIGHT = 2,
    X8R_BOUNDARY_AUTO = 3,
} x8r_boundary_mode;

typedef enum {
    X8R_CUT_EOF = 0,
    X8R_CUT_HARD = 1,
    X8R_CUT_LINE = 2,
    X8R_CUT_BLANK = 3,
    X8R_CUT_BLOCK = 4,
} x8r_cut_kind;

typedef struct {
    size_t start;
    size_t end;
    size_t token_count;
    x8r_cut_kind cut;
} x8r_chunk;

typedef struct {
    size_t budget;
    x8r_vocab_id vocab;
    x8r_boundary_mode boundary;
    double tolerance;
} x8r_opts;

typedef struct x8r_ctx x8r_ctx;

x8r_status x8r_ctx_open(const char *vocab_path, x8r_vocab_id vocab, x8r_ctx **out);
void x8r_ctx_close(x8r_ctx *ctx);

size_t x8r_count_tokens(x8r_ctx *ctx, const uint8_t *buf, size_t len);

/* encode buf into a freshly-allocated array of token ids.
 * caller frees with x8r_ids_free.
 * matches tiktoken.Encoding.encode_ordinary bit-for-bit.
 */
x8r_status x8r_encode_ordinary(x8r_ctx *ctx,
                               const uint8_t *buf, size_t len,
                               uint32_t **out_ids, size_t *out_n);
void x8r_ids_free(uint32_t *ids);

/* decode a sequence of token ids back to the original byte stream.
 * caller frees with x8r_bytes_free.
 * returns X8R_E_VOCAB if any id is unknown.
 * matches tiktoken.Encoding.decode_bytes for the same id list.
 */
x8r_status x8r_decode_bytes(x8r_ctx *ctx,
                            const uint32_t *ids, size_t n,
                            uint8_t **out_bytes, size_t *out_len);
void x8r_bytes_free(uint8_t *bytes);

/* per-id decode for callers building their own buffers. returns NULL if
 * the id is unknown to the loaded vocab. *out_len is set to the byte
 * length of the token. the returned pointer is borrowed from the vocab
 * mmap and is valid until x8r_ctx_close.
 */
const uint8_t *x8r_decode_single(x8r_ctx *ctx, uint32_t id, size_t *out_len);

x8r_status x8r_chunk_buf(x8r_ctx *ctx,
                         const uint8_t *buf, size_t len,
                         const x8r_opts *opts,
                         x8r_chunk **out_chunks, size_t *out_n);

void x8r_chunks_free(x8r_chunk *chunks);

const char *x8r_version(void);

#ifdef __cplusplus
}
#endif
#endif
