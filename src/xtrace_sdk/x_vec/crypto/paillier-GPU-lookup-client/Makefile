# GPU_paillier_client/Makefile
#
# Build a Python extension module paillier_GPU_lookup_client.* for different GPU archs and key sizes.
#
# Usage examples:
#   make                                           # defaults: KEY_BITS=1024, SMS="70 80 90", TPI=32
#   make KEY_BITS=2048 SMS="80 90"                 # compile for A100/H100 with 8192-bit n^2
#   make KEY_BITS=1024 TPI=32 SMS="86"             # compile for RTX 30-series (sm_86)
#
# Directory layout (as you described):
#   ./lookup_GPU_paillier_client/paillier_GPU_lookup_client.cu
#   ./GPU_include/pybind11/...
#   ./GPU_include/cgbn/...

# -------- Parameters you can tweak on the command line --------
KEY_BITS  ?= 1024
TPI       ?= 32
ALPHA_LEN ?= 280
# List of SM versions to build for (space separated)
SMS      ?= 70 80 86 89 90

# Compiler/paths
NVCC     ?= nvcc
CUDA_PATH?= /usr/local/cuda

# Python extension suffix (e.g., .cpython-311-x86_64-linux-gnu.so)
PY_INC   := $(shell python3 -c 'import sysconfig, sys; print(sysconfig.get_paths()["include"])')
EXT      := $(shell python3 -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
TARGET   := paillier_GPU_lookup_client$(EXT)

SRC      := paillier_GPU_lookup_client.cu
INCDIR   := ../GPU_include
CXXSTD   := -std=c++17

# Construct gencode flags for each SM
GENCODE  := $(foreach sm,$(SMS),-gencode arch=compute_$(sm),code=sm_$(sm))

# Include paths
INCLUDES := -I$(INCDIR) -I$(INCDIR)/pybind11 -I$(INCDIR)/cgbn -I$(CUDA_PATH)/include -I$(PY_INC)

# Build flags
NVCCFLAGS := $(CXXSTD) -O3 -Xcompiler -fPIC -shared \
             -DKEY_BITS=$(KEY_BITS) -DALPHA_LEN=$(ALPHA_LEN) -DCGBN_TPI=$(TPI) -DCGBN_USE_GMP \
             -Wno-deprecated-gpu-targets \
             $(GENCODE) $(INCLUDES)

# Link GMP for the CPU fallback (header demands GMP present)
LDLIBS := -lgmp

# Note: pybind11 is header-only; normally you don't need to link libpython explicitly.
# If your system requires it, uncomment the following:
# PY_LDFLAGS := $(shell python3-config --ldflags)
# LDFLAGS    := $(PY_LDFLAGS)

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f $(TARGET)
