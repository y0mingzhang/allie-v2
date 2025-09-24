

# git clone https://github.com/NVIDIA/apex
cd apex

# To build all contrib extensions at once
NVCC_APPEND_FLAGS="--threads 16" APEX_PARALLEL_BUILD=32 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_ALL_CONTRIB_EXT=1 uv pip install -v --no-build-isolation .
