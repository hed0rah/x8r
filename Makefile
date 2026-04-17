CC      ?= gcc
CFLAGS  ?= -std=c11 -O3 -march=x86-64-v3 -Wall -Wextra -Wpedantic -Wno-unused-parameter
CPPFLAGS = -Iinclude -Isrc -D_POSIX_C_SOURCE=200809L
LDFLAGS ?=

BUILD   := build
LIB_SRC := src/mmap_io.c src/vocab.c src/pretok_scalar.c src/pretok_avx2.c src/bpe.c src/chunk.c
LIB_OBJ := $(LIB_SRC:src/%.c=$(BUILD)/%.o)
LIB_PIC := $(LIB_SRC:src/%.c=$(BUILD)/%.pic.o)

CLI_BIN := $(BUILD)/x8r
LIB_SO  := $(BUILD)/libx8r.so
LIB_A   := $(BUILD)/libx8r.a

all: $(CLI_BIN) $(LIB_SO) $(LIB_A)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.c | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD)/%.pic.o: src/%.c | $(BUILD)
	$(CC) $(CFLAGS) $(CPPFLAGS) -fPIC -c $< -o $@

$(CLI_BIN): $(LIB_OBJ) $(BUILD)/main.o
	$(CC) $(CFLAGS) $(LIB_OBJ) $(BUILD)/main.o -o $@ $(LDFLAGS)

$(LIB_A): $(LIB_OBJ)
	ar rcs $@ $^

$(LIB_SO): $(LIB_PIC)
	$(CC) $(CFLAGS) -shared $(LIB_PIC) -o $@ $(LDFLAGS) -Wl,-soname,libx8r.so

clean:
	rm -rf $(BUILD)

.PHONY: all clean
