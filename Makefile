CC      = clang
CFLAGS  = -O2 -Wall -Wextra -std=c11
FW      = -framework Metal -framework MetalPerformanceShaders \
          -framework Foundation -framework Accelerate
BIN     = bin

$(BIN):
	mkdir -p $(BIN)

# Metal shader compilation
shaders.metallib: src/shaders.metal
	xcrun -sdk macosx metal -c src/shaders.metal -o /tmp/shaders.air
	xcrun -sdk macosx metallib /tmp/shaders.air -o shaders.metallib

# Tests — CPU (default DEVICE="cpu")
test: $(BIN) test/test_term.m src/tinyhvm.c src/tinyhvm.h
	$(CC) $(CFLAGS) $(FW) -o $(BIN)/test_term test/test_term.m && $(BIN)/test_term

# Tests — Metal
test_metal: shaders.metallib $(BIN) test/test_term.m src/tinyhvm.c src/tinyhvm.h
	$(CC) $(CFLAGS) $(FW) -DDEVICE='"metal"' -o $(BIN)/test_term test/test_term.m && $(BIN)/test_term

# Training — CPU
test_train: $(BIN) test/test_train.m src/tinyhvm.c src/tinyhvm.h
	$(CC) $(CFLAGS) $(FW) -o $(BIN)/test_train test/test_train.m && $(BIN)/test_train

# Training — Metal
test_train_metal: shaders.metallib $(BIN) test/test_train.m src/tinyhvm.c src/tinyhvm.h
	$(CC) $(CFLAGS) $(FW) -DDEVICE='"metal"' -o $(BIN)/test_train test/test_train.m && $(BIN)/test_train

# Tensor NN API parity (cpu + metal)
test_tensor_nn_layers: shaders.metallib $(BIN) test/test_tensor_nn_layers.m src/tinyhvm.c src/tinyhvm.h
	$(CC) $(CFLAGS) $(FW) -o $(BIN)/test_tensor_nn_layers test/test_tensor_nn_layers.m && $(BIN)/test_tensor_nn_layers

# Tensor NN backward checks (cpu + metal)
test_tensor_nn_backward: shaders.metallib $(BIN) test/test_tensor_nn_backward.m src/tinyhvm.c src/tinyhvm.h
	$(CC) $(CFLAGS) $(FW) -o $(BIN)/test_tensor_nn_backward test/test_tensor_nn_backward.m && $(BIN)/test_tensor_nn_backward

clean:
	rm -rf $(BIN) shaders.metallib
	rm -rf *.dSYM

# README snippet regression (forward + autograd bundle); see test/test_readme_verify.m
readme-verify: $(BIN) shaders.metallib test/test_readme_verify.m src/tinyhvm.c src/tinyhvm.h
	$(CC) $(CFLAGS) $(FW) -DDEVICE='"cpu"' -o $(BIN)/test_readme_verify test/test_readme_verify.m && $(BIN)/test_readme_verify
	$(CC) $(CFLAGS) $(FW) -DDEVICE='"metal"' -o $(BIN)/test_readme_verify test/test_readme_verify.m && $(BIN)/test_readme_verify

.PHONY: test test_metal test_train test_train_metal test_tensor_nn_layers test_tensor_nn_backward readme-verify clean
