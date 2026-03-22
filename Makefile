CC      = clang
CFLAGS  = -O2 -Wall -Wextra -std=c11
FW      = -framework Metal -framework MetalPerformanceShaders \
          -framework Foundation -framework Accelerate

# Metal shader compilation
shaders.metallib: src/shaders.metal
	xcrun -sdk macosx metal -c src/shaders.metal -o /tmp/shaders.air
	xcrun -sdk macosx metallib /tmp/shaders.air -o shaders.metallib

# Tests — CPU (default DEVICE="cpu")
test: test/test_term.m src/tinyhvm.c src/tinyhvm.h src/gpu_cpu.c src/gpu_metal.m
	$(CC) $(CFLAGS) $(FW) -o test_term test/test_term.m && ./test_term

# Tests — Metal (override DEVICE)
test_metal: shaders.metallib test/test_term.m src/tinyhvm.c src/tinyhvm.h src/gpu_cpu.c src/gpu_metal.m
	$(CC) $(CFLAGS) $(FW) -DDEVICE='"metal"' -o test_term test/test_term.m && ./test_term

# Training — CPU
test_train: test/test_train.m src/tinyhvm.c src/tinyhvm.h src/gpu_cpu.c src/gpu_metal.m
	$(CC) $(CFLAGS) $(FW) -o test_train test/test_train.m && ./test_train

# Training — Metal
test_train_metal: shaders.metallib test/test_train.m src/tinyhvm.c src/tinyhvm.h src/gpu_cpu.c src/gpu_metal.m
	$(CC) $(CFLAGS) $(FW) -DDEVICE='"metal"' -o test_train test/test_train.m && ./test_train

clean:
	rm -f test_term test_train shaders.metallib
	rm -rf *.dSYM

.PHONY: test test_metal test_train test_train_metal clean
