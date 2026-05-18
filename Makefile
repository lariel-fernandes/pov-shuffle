# Makefile for standalone CUDA module compilation

# Compiler and tools
NVCC ?= nvcc
PYTHON ?= .venv/bin/python

# Directories
SRC_DIR ?= src/povs/__cuda/lib
BUILD_DIR ?= build

# Auto-detect CUDA architecture from nvidia-smi, fallback to sm_75
DETECTED_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')
CUDA_ARCH ?= $(if $(DETECTED_ARCH),sm_$(DETECTED_ARCH),sm_75)

# Get PyTorch include paths
TORCH_INCLUDE_DIR := $(shell $(PYTHON) -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'include'))" 2>/dev/null)
TORCH_INCLUDES := $(if $(TORCH_INCLUDE_DIR),-I$(TORCH_INCLUDE_DIR) -I$(TORCH_INCLUDE_DIR)/torch/csrc/api/include,)

# Get CUTLASS include paths
CUTLASS_INCLUDE_DIR := $(shell $(PYTHON) -c "import cutlass_library; import os; print(os.path.join(os.path.dirname(cutlass_library.__file__), 'source', 'include'))" 2>/dev/null)
CUTLASS_INCLUDES := $(if $(CUTLASS_INCLUDE_DIR),-I$(CUTLASS_INCLUDE_DIR),)

# Compiler flags
NVCC_FLAGS ?= -std=c++17 -O2 --expt-relaxed-constexpr
NVCC_DEBUG_FLAGS ?= -std=c++17 -g -lineinfo -O1 --expt-relaxed-constexpr
CUDA_FLAGS ?= -arch=$(CUDA_ARCH) -DWITH_CUDA -DSTANDALONE_BUILD
INCLUDES ?= -I$(SRC_DIR) $(TORCH_INCLUDES) $(CUTLASS_INCLUDES)

# Source and target
PROGRAM ?= povs
PROGRAM_H := $(SRC_DIR)/$(PROGRAM).h
PROGRAM_CU := $(SRC_DIR)/$(PROGRAM).cu
PROGRAM_BIN := $(BUILD_DIR)/$(PROGRAM).bin
UTILS_H := $(SRC_DIR)/utils.h
UTILS_CU := $(SRC_DIR)/utils.cu

$(PROGRAM_BIN): $(PROGRAM_CU) $(PROGRAM_H) $(UTILS_CU) $(UTILS_H)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_FLAGS) $(INCLUDES) -o $@ $< $(UTILS_CU)

.PHONY: compile
compile: $(PROGRAM_BIN)

.PHONY: debug
debug:
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_DEBUG_FLAGS) $(CUDA_FLAGS) $(INCLUDES) -o $(PROGRAM_BIN) $(PROGRAM_CU) $(UTILS_CU)

.PHONY: run
run:
	@$(PROGRAM_BIN)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all
all: clean compile

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  compile - Build program (optimized)"
	@echo "  debug   - Build program with debug symbols (-g -G -O0)"
	@echo "  run     - Run program"
	@echo "  clean   - Remove build artifacts"
	@echo "  all     - Clean and compile (default)"
	@echo ""
	@echo "Variables (can override):"
	@echo "  NVCC        = $(NVCC)"
	@echo "  PYTHON      = $(PYTHON)"
	@echo "  CUDA_ARCH   = $(CUDA_ARCH)"
	@echo "  SRC_DIR     = $(SRC_DIR)"
	@echo "  PROGRAM     = $(PROGRAM)"
