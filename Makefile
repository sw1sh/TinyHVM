CC      = clang
CFLAGS  = -O2 -Wall -Wextra -Wpedantic -std=c11
LDFLAGS =

SRC     = src/tinyhvm.c
HDR     = src/tinyhvm.h

# CPU backend uses Accelerate for BLAS
cpu: $(SRC) $(HDR) src/gpu_cpu.c
	$(CC) $(CFLAGS) -DBACKEND_CPU -framework Accelerate \
		-o tinyhvm src/main.c src/gpu_cpu.c $(LDFLAGS)

# Metal backend (macOS)
metal: $(SRC) $(HDR) src/gpu_metal.m
	$(CC) $(CFLAGS) -DBACKEND_METAL \
		-framework Metal -framework MetalPerformanceShaders -framework Foundation \
		-o tinyhvm src/main.c src/gpu_metal.m $(LDFLAGS)

# Tests
test_term: test/test_term.c $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o test_term test/test_term.c
	./test_term

test: test_term

clean:
	rm -f tinyhvm tinyhvm_cpu test_term test_reduce test_basic
	rm -rf *.dSYM

.PHONY: cpu metal test test_term clean
