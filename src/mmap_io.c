#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#include "internal.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

x8r_status x8r_mmap_ro(const char *path, const uint8_t **out_buf, size_t *out_len, void **out_handle) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return X8R_E_IO;

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return X8R_E_IO;
    }

    if (st.st_size == 0) {
        close(fd);
        *out_buf = (const uint8_t *)"";
        *out_len = 0;
        *out_handle = NULL;
        return X8R_OK;
    }

    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return X8R_E_IO;

    madvise(p, (size_t)st.st_size, MADV_SEQUENTIAL);
    madvise(p, (size_t)st.st_size, MADV_WILLNEED);

    *out_buf = (const uint8_t *)p;
    *out_len = (size_t)st.st_size;
    *out_handle = p;
    return X8R_OK;
}

void x8r_munmap(void *handle, size_t len) {
    if (handle && len) munmap(handle, len);
}
