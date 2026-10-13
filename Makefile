# Lab 6 build.
#   make            build everything into build/
#   make cpu        only the CPU programs (no CUDA needed)
#   make ARCH=-arch=sm_86   if your nvcc is older than 11.6 (no -arch=native);
#                           sm_75 = RTX 20xx/T4, sm_86 = RTX 30xx, sm_89 = RTX 40xx,
#                           sm_120 = RTX 50xx (needs CUDA 12.8+)
NVCC    ?= nvcc
CC      ?= gcc
ARCH    ?= -arch=native
NVFLAGS ?= -O3 $(ARCH) -std=c++17 -lineinfo
CFLAGS  ?= -O2
B       := build
S       := src

GPU_BINS := $(B)/matrix_gpu $(B)/matrix_tiled $(B)/matrix_tiled32 $(B)/matrix_cublas \
            $(B)/matrix_opt $(B)/conv_gpu
CPU_BINS := $(B)/matrix_cpu $(B)/conv_cpu

.PHONY: all cpu gpu lib clean
all: cpu gpu lib
cpu: $(CPU_BINS)
gpu: $(GPU_BINS)
lib: $(B)/libmatrix.so

$(B):
	mkdir -p $(B)

$(B)/matrix_cpu: $(S)/matrix_cpu.c | $(B)
	$(CC) $(CFLAGS) $< -o $@

$(B)/conv_cpu: $(S)/conv_cpu.c $(S)/conv_io.h | $(B)
	$(CC) $(CFLAGS) $< -o $@

$(B)/matrix_gpu: $(S)/matrix_gpu.cu $(S)/common.cuh | $(B)
	$(NVCC) $(NVFLAGS) $< -o $@

$(B)/matrix_tiled: $(S)/matrix_tiled.cu $(S)/common.cuh | $(B)
	$(NVCC) $(NVFLAGS) $< -o $@

# Same kernel with 32x32 tiles (1024 threads/block) — for the tile-size trade-off discussion.
$(B)/matrix_tiled32: $(S)/matrix_tiled.cu $(S)/common.cuh | $(B)
	$(NVCC) $(NVFLAGS) -DTILE_WIDTH=32 -DIMPL_NAME='"tiled32_cuda"' $< -o $@

$(B)/matrix_cublas: $(S)/matrix_cublas.cu $(S)/common.cuh | $(B)
	$(NVCC) $(NVFLAGS) $< -o $@ -lcublas

$(B)/matrix_opt: $(S)/matrix_opt.cu $(S)/sgemm_opt.cuh $(S)/common.cuh | $(B)
	$(NVCC) $(NVFLAGS) $< -o $@

$(B)/conv_gpu: $(S)/conv_gpu.cu $(S)/conv_kernels.cuh $(S)/conv_io.h | $(B)
	$(NVCC) $(NVFLAGS) $< -o $@

$(B)/libmatrix.so: $(S)/matrix_lib.cu $(S)/sgemm_opt.cuh $(S)/conv_kernels.cuh | $(B)
	$(NVCC) $(NVFLAGS) -Xcompiler -fPIC -shared $< -o $@ -lcublas

clean:
	rm -rf $(B)
