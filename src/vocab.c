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
        if (elen == (uint16_t)len && memcmp(ebytes, bytes, len) == 0) return erank;
        i = (i + 1) & mask;
    }
    return UINT32_MAX;
}
