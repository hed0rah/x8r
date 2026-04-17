#ifndef X8R_INTERNAL_H
#define X8R_INTERNAL_H

#include <stddef.h>
#include <stdint.h>
#include "x8r.h"

/*
 * on-disk vocab file layout, little-endian:
 *
 *   magic        "X8RV"           4 bytes
 *   version      u32              1
 *   vocab_id     u32              (x8r_vocab_id)
 *   n_tokens     u32              total number of tokens
 *   table_size   u32              power of 2, hash table capacity
 *   data_bytes   u32              total bytes in token blob
 *   reserved     u32              0
 *   table[table_size]   u32       offset into data blob, or 0xFFFFFFFF for empty
 *                                 slot maps hash -> data offset of (len:u16, rank:u32, bytes[len])
 *   data[data_bytes]   u8         packed entries
 *
 * every entry in data is: u16 len | u32 rank | u8 bytes[len]
 * stored open-addressed with linear probing.
 */

#define X8R_VOCAB_MAGIC "X8RV"
#define X8R_VOCAB_VERSION 1u
#define X8R_VOCAB_EMPTY 0xFFFFFFFFu

typedef struct {
    uint32_t n_tokens;
    uint32_t table_size;     /* power of 2 */
    uint32_t table_mask;
    uint32_t data_bytes;
    const uint32_t *table;   /* table_size entries */
    const uint8_t *data;     /* raw blob */
    void *map_base;          /* for munmap */
    size_t map_size;
} x8r_vocab;

struct x8r_ctx {
    x8r_vocab vocab;
    x8r_vocab_id id;
};

/* mmap_io.c */
x8r_status x8r_mmap_ro(const char *path, const uint8_t **out_buf, size_t *out_len, void **out_handle);
void x8r_munmap(void *handle, size_t len);

/* vocab.c */
x8r_status x8r_vocab_load(const char *path, x8r_vocab *out);
void x8r_vocab_close(x8r_vocab *v);
/* lookup: returns rank for token bytes, or UINT32_MAX if missing */
uint32_t x8r_vocab_lookup(const x8r_vocab *v, const uint8_t *bytes, size_t len);

/* pretok_scalar.c
 * splits input into pre-tokens per the cl100k regex. calls `sink` for each.
 * returns number of pre-tokens emitted.
 */
typedef void (*x8r_pretok_sink)(void *user, size_t start, size_t len);
size_t x8r_pretokenize_scalar(const uint8_t *buf, size_t len,
                              x8r_pretok_sink sink, void *user);

/* pretok_avx2.c: ASCII-only accelerators. return the first position at
 * or after p that either (a) isn't in the target class, or (b) is a
 * non-ASCII byte (caller falls back to scalar for that byte). */
size_t x8r_scan_ascii_letter(const uint8_t *b, size_t n, size_t p);
size_t x8r_scan_ascii_space(const uint8_t *b, size_t n, size_t p);

/* bpe.c
 * encode a single pre-token to token ranks, append to *ranks_out.
 * returns number of tokens emitted.
 */
size_t x8r_bpe_encode(const x8r_vocab *v,
                      const uint8_t *bytes, size_t len,
                      uint32_t **ranks_out, size_t *ranks_cap, size_t *ranks_len);

/* hashing: must match the hash used by the vocab generator (scripts/dump_cl100k.py) */
static inline uint32_t x8r_hash_bytes(const uint8_t *b, size_t n) {
    /* fnv-1a 32-bit */
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 16777619u;
    }
    return h;
}

#endif
