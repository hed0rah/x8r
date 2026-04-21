#define _POSIX_C_SOURCE 200809L
#include "internal.h"

#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>

static inline void vocab_entry_read(const uint8_t *data, uint32_t off,
                                    uint16_t *out_len, uint32_t *out_rank,
                                    const uint8_t **out_bytes) {
    uint16_t len;
    uint32_t rank;
    memcpy(&len, data + off, 2);
    memcpy(&rank, data + off + 2, 4);
    *out_len = len;
    *out_rank = rank;
    *out_bytes = data + off + 6;
}

/* equal-length byte compare, specialized for short keys.
 *
 * cl100k length distribution: 79% of tokens are <= 8 bytes, 99% <= 16.
 * libc's __memcmp_avx2_movbe previously cost ~6-8% of total runtime in the
 * perf profile, almost entirely call/setup overhead for these tiny compares.
 * a switch on len lets the compiler emit 1-2 mov+xor+jnz for the hot path
 * and never call libc for len <= 16.
 *
 * unaligned loads are fine on x86; gcc lowers memcpy(&u64, p, 8) to a
 * single movq. the vocab blob has no alignment guarantees past byte 0. */
static inline int bytes_eq_fast(const uint8_t *a, const uint8_t *b, size_t len) {
    uint64_t a0, a1, b0, b1;
    switch (len) {
    case 0: return 1;
    case 1: return a[0] == b[0];
    case 2: {
        uint16_t x, y;
        memcpy(&x, a, 2); memcpy(&y, b, 2);
        return x == y;
    }
    case 3: {
        uint16_t x, y;
        memcpy(&x, a, 2); memcpy(&y, b, 2);
        return x == y && a[2] == b[2];
    }
    case 4: {
        uint32_t x, y;
        memcpy(&x, a, 4); memcpy(&y, b, 4);
        return x == y;
    }
    case 5: case 6: case 7: {
        /* overlapping 4-byte loads cover [0..3] and [len-4..len-1]; every
         * byte gets read at least once, no out-of-bounds access. */
        uint32_t x0, x1, y0, y1;
        memcpy(&x0, a, 4);             memcpy(&y0, b, 4);
        memcpy(&x1, a + len - 4, 4);   memcpy(&y1, b + len - 4, 4);
        return x0 == y0 && x1 == y1;
    }
    case 8:
        memcpy(&a0, a, 8); memcpy(&b0, b, 8);
        return a0 == b0;
    case 9: case 10: case 11: case 12:
    case 13: case 14: case 15: case 16:
        memcpy(&a0, a, 8);             memcpy(&b0, b, 8);
        memcpy(&a1, a + len - 8, 8);   memcpy(&b1, b + len - 8, 8);
        return a0 == b0 && a1 == b1;
    default:
        return memcmp(a, b, len) == 0;
    }
}

x8r_status x8r_vocab_load(const char *path, x8r_vocab *out) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return X8R_E_VOCAB;

    struct stat st;
    if (fstat(fd, &st) != 0 || (size_t)st.st_size < 32) {
        close(fd);
        return X8R_E_VOCAB;
    }

    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return X8R_E_VOCAB;

    const uint8_t *b = (const uint8_t *)p;
    if (memcmp(b, X8R_VOCAB_MAGIC, 4) != 0) {
        munmap(p, (size_t)st.st_size);
        return X8R_E_VOCAB;
    }

    uint32_t ver, vid, n, tsz, dbytes, reserved;
    memcpy(&ver,      b + 4,  4);
    memcpy(&vid,      b + 8,  4);
    memcpy(&n,        b + 12, 4);
    memcpy(&tsz,      b + 16, 4);
    memcpy(&dbytes,   b + 20, 4);
    memcpy(&reserved, b + 24, 4);
    (void)reserved;

    if (ver != X8R_VOCAB_VERSION) {
        munmap(p, (size_t)st.st_size);
        return X8R_E_VOCAB;
    }
    if ((tsz & (tsz - 1)) != 0 || tsz == 0) {
        munmap(p, (size_t)st.st_size);
        return X8R_E_VOCAB;
    }

    const size_t header = 28;
    if ((size_t)st.st_size < header + (size_t)tsz * 4 + dbytes) {
        munmap(p, (size_t)st.st_size);
        return X8R_E_VOCAB;
    }

    out->n_tokens = n;
    out->table_size = tsz;
    out->table_mask = tsz - 1;
    out->data_bytes = dbytes;
    out->vocab_id = vid;
    out->table = (const uint32_t *)(b + header);
    out->data  = b + header + (size_t)tsz * 4;
    out->map_base = p;
    out->map_size = (size_t)st.st_size;
    return X8R_OK;
}

void x8r_vocab_close(x8r_vocab *v) {
    if (v && v->map_base) {
        munmap(v->map_base, v->map_size);
        v->map_base = NULL;
    }
}

uint32_t x8r_vocab_lookup(const x8r_vocab *v, const uint8_t *bytes, size_t len) {
    if (len > 0xFFFF) return UINT32_MAX;
    uint32_t h = x8r_hash_bytes(bytes, len);
    uint32_t mask = v->table_mask;
    uint32_t i = h & mask;
    const uint32_t *table = v->table;
    const uint8_t *data = v->data;
    for (uint32_t probe = 0; probe < v->table_size; ++probe) {
        uint32_t off = table[i];
        /* prefetch the next probe slot in case of collision; cheap because
         * the same cache line often holds it (8 slots per 64B line). */
        __builtin_prefetch(&table[(i + 1) & mask], 0, 0);
        if (off == X8R_VOCAB_EMPTY) return UINT32_MAX;
        /* prefetch the entry header before we touch it */
        __builtin_prefetch(data + off, 0, 0);
        uint16_t elen;
        uint32_t erank;
        const uint8_t *ebytes;
        vocab_entry_read(data, off, &elen, &erank, &ebytes);
        if (elen == (uint16_t)len && bytes_eq_fast(ebytes, bytes, len)) return erank;
        i = (i + 1) & mask;
    }
    return UINT32_MAX;
}
