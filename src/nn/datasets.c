// nn/datasets.c — MNIST dataset loader
// Usage: MNISTData data = mnist_load("data");

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    f32 *train_images;   // [n_train, 784] raw [0,255]
    u8  *train_labels;   // [n_train]
    f32 *test_images;    // [n_test, 784]
    u8  *test_labels;    // [n_test]
    u32 n_train, n_test;
} MNISTData;

static u32 read_u32_be(FILE *f) {
    u8 buf[4]; fread(buf, 1, 4, f);
    return ((u32)buf[0]<<24) | ((u32)buf[1]<<16) | ((u32)buf[2]<<8) | buf[3];
}

static f32 *mnist_load_images(const char *path, u32 *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); exit(1); }
    read_u32_be(f); *n = read_u32_be(f);
    u32 rows = read_u32_be(f), cols = read_u32_be(f);
    u32 sz = *n * rows * cols;
    u8 *raw = malloc(sz);
    fread(raw, 1, sz, f); fclose(f);
    f32 *out = malloc(sz * sizeof(f32));
    for (u32 i = 0; i < sz; i++) out[i] = (f32)raw[i];  // raw [0,255] matching tinygrad
    free(raw);
    return out;
}

static u8 *mnist_load_labels(const char *path, u32 *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); exit(1); }
    read_u32_be(f); *n = read_u32_be(f);
    u8 *out = malloc(*n);
    fread(out, 1, *n, f); fclose(f);
    return out;
}

static MNISTData mnist_load(const char *dir) {
    MNISTData d = {0};
    char path[256];
    snprintf(path, sizeof(path), "%s/train-images-idx3-ubyte", dir);
    d.train_images = mnist_load_images(path, &d.n_train);
    snprintf(path, sizeof(path), "%s/train-labels-idx1-ubyte", dir);
    d.train_labels = mnist_load_labels(path, &d.n_train);
    snprintf(path, sizeof(path), "%s/t10k-images-idx3-ubyte", dir);
    d.test_images = mnist_load_images(path, &d.n_test);
    snprintf(path, sizeof(path), "%s/t10k-labels-idx1-ubyte", dir);
    d.test_labels = mnist_load_labels(path, &d.n_test);
    return d;
}

static void mnist_free(MNISTData *d) {
    free(d->train_images); free(d->train_labels);
    free(d->test_images);  free(d->test_labels);
}
