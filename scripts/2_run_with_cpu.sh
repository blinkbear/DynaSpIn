cd build
cmake .. -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DLLAMA_MPI=1
cmake --build . --target speculative --config Release
cd .. 
DRAFT_MODEL_PATH="/home/wychen/.cache/huggingface/hub/llama-68m/ggml-model-f16.gguf"
TARGET_MODEL_PATH="/home/wychen/.cache/huggingface/hub/llama-2-13b/ggml-model-f16.gguf"
PROMPT_PATH="./prompts/parallel-questions.txt"
HOSTS_PATH="./hosts"
mpirun -c 6 -x NCCL_SOCKET_IFNAME=eno12399np0 -x NCCL_IB_DISABLE=0 -x NCCL_IB_CUDA_SUPPORT=1 -mca btl_tcp_if_include eno12399np0  -hostfile ${HOSTS_PATH} --bind-to none \
    build/bin/speculative \
    -md ${DRAFT_MODEL_PATH} \
    -m ${TARGET_MODEL_PATH}  \
    -e \
    -f ${PROMPT_PATH} \
    -n 128 \
    --mpi-layer-split 0.2,0.2,0.2,0.2,0.2/1.0 \
    --ignore-eos \
    --temp -1.0 \
    --repeat-last-n 0 \
    --draft 4 \
    -tb 32,32 \
    -t 32,32 \
    -c 1024 \
    -pa 0.001 \
    -ps 0.8 \
    --numa \
    -pr 0.4 \
    -pd 0.01 \
    -np 3 2>&1 | tee ./result.log