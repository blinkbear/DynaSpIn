# CMAKE_ARGS="-DLLAMA_CUBLAS=on"
# FORCE_CMAKE=1
# pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LLAMA_CUBLAS=on

cd build

# 启用CUDA支持，并确保使用mpicc和mpicxx编译器
cmake .. -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DLLAMA_MPI=1 -DCUDA=1 -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS

# 构建项目，确保Release配置下启用了CUDA
cmake --build . --target speculative --config Release

cd ..

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export ORTE_BASE_USER_DEBUGGER=/usr/bin/gdb
export NVIDIA_VISIBLE_DEVICES=0,1
NP_PER_NODE=2
TOTAL_NP=4

DRAFT_MODEL_PATH="/home/wychen/.cache/huggingface/hub/llama-160m/ggml-model-f16.gguf"
TARGET_MODEL_PATH="/home/wychen/.cache/huggingface/hub/llama-2-13b/ggml-model-f16.gguf"
HOSTS_PATH="./hosts"
PROMPT_PATH="./prompts/mtbench_prompts.json"

jq -c '.[]' $PROMPT_PATH | while read item; do
    id=$(echo "$item" | jq -r '.[0]')
    prompt=$(echo "$item" | jq -r '.[1]')

    echo "Running with id: $id"
    # prompt=$(echo "$prompt" | sed -E 's/\[INST\]//g; s/\[\/INST\]//g; s/nnASSISTANT:?//g')
    echo "Prompt: $prompt"

    mpirun --allow-run-as-root -np $TOTAL_NP --hostfile ${HOSTS_PATH} --bind-to none --map-by ppr:$NP_PER_NODE:node --report-bindings -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=eno12399np0 -x NCCL_IB_DISABLE=0 -x NCCL_IB_CUDA_SUPPORT=1 -mca btl_tcp_if_include eno12399np0 \
    build/bin/speculative \
    -md ${DRAFT_MODEL_PATH} \
    -m ${TARGET_MODEL_PATH}  \
    -e \
    -n 10 \
    --mpi-layer-split 0.5,0.3,0.2/1.0 \
    --ignore-eos \
    --temp -1.0 \
    --repeat-last-n 0 \
    --draft 8 \
    --threads-batch 32 \
    --threads 32 \
    --ctx-size 1024 \
    --p-accept 0 \
    --p-split 0.001 \
    --p-recovery 0.4 \
    --p-decay 0.01 \
    --batch-size 256 \
    --cont-batching \
    --parallel 3 \
    --prompt "$prompt"  >> ./result.log  
done