CC      = clang
CFLAGS  = -O2 -Wall -Wextra -std=c11
FRAMEWORKS_CPU   = -framework Accelerate
FRAMEWORKS_METAL = -framework Metal -framework MetalPerformanceShaders \
                   -framework Foundation -framework Accelerate

# Build shaders.metallib from shaders.metal
shaders.metallib: src/shaders.metal
	xcrun -sdk macosx metal -c src/shaders.metal -o /tmp/shaders.air
	xcrun -sdk macosx metallib /tmp/shaders.air -o shaders.metallib

# Tests — CPU backend (header-only include)
test: test/test_term.c src/tinyhvm.c src/tinyhvm.h src/gpu_cpu.c
	$(CC) $(CFLAGS) $(FRAMEWORKS_CPU) -o test_term test/test_term.c && ./test_term

# Tests — Metal backend parity
test_metal: shaders.metallib test/test_metal.c src/tinyhvm.c src/tinyhvm.h src/gpu_metal.m
	$(CC) $(CFLAGS) $(FRAMEWORKS_METAL) -o test_metal test/test_metal.c src/gpu_metal.m && ./test_metal

# Training — CPU
test_train: test/test_train.c src/tinyhvm.c src/tinyhvm.h src/gpu_cpu.c
	$(CC) $(CFLAGS) $(FRAMEWORKS_CPU) -o test_train test/test_train.c && ./test_train

# Training — Metal
test_train_metal: shaders.metallib test/test_train_metal.c src/tinyhvm.c src/tinyhvm.h src/gpu_metal.m
	$(CC) $(CFLAGS) $(FRAMEWORKS_METAL) -o test_train_metal test/test_train_metal.c src/gpu_metal.m && ./test_train_metal

clean:
	rm -f test_term test_metal test_train test_train_metal shaders.metallib /tmp/shaders.air
	rm -rf *.dSYM

.PHONY: test test_metal test_train test_train_metal clean
