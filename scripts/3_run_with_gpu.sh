cd build

# 启用CUDA支持，并确保使用mpicc和mpicxx编译器
cmake .. -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DLLAMA_MPI=1 -DCUDA=1

# 构建项目，确保Release配置下启用了CUDA
cmake --build . --target speculative --config Release

cd ..

export ORTE_BASE_USER_DEBUGGER=/usr/bin/gdb
export NVIDIA_VISIBLE_DEVICES=0,1
NP_PER_NODE=2
TOTAL_NP=4

DRAFT_MODEL_PATH="/home/wychen/.cache/huggingface/hub/llama-68m/ggml-model-f16.gguf"
TARGET_MODEL_PATH="/home/wychen/.cache/huggingface/hub/llama-2-13b/ggml-model-f16.gguf"
PROMPT_PATH="./prompts/dan.txt"
HOSTS_PATH="./hosts"

mpirun -np $TOTAL_NP --hostfile ${HOSTS_PATH} --bind-to none --map-by ppr:$NP_PER_NODE:node --report-bindings -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=eno12399np0 -x NCCL_IB_DISABLE=0 -x NCCL_IB_CUDA_SUPPORT=1 -mca btl_tcp_if_include eno12399np0 \
    build/bin/speculative \
    -md ${DRAFT_MODEL_PATH} \
    -m ${TARGET_MODEL_PATH}  \
    -e \
    -f ${PROMPT_PATH} \
    -n 128 \
    --mpi-layer-split 0.25,0.25,0.25,0.25/1.0 \
    --ignore-eos \
    --temp -1.0 \
    --repeat-last-n 0 \
    --draft 4 \
    -tb 32,32 \
    -t 32,32 \
    -c 1024 \
    -pa 0.001 \
    -ps 0.8 \
    -pr 0.4 \
    -pd 0.01 \
    -np 3 2>&1 | tee ./result.log