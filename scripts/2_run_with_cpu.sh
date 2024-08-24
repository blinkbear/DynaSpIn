cd build
cmake .. -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DLLAMA_MPI=1
cmake --build . --target speculative --config Release
cd .. 
USER_NAME=$(whoami)
DRAFT_MODEL_PATH="/home/${USER_NAME}/.cache/huggingface/hub/llama-160m/ggml-model-f16.gguf"
TARGET_MODEL_PATH="/home/${USER_NAME}/.cache/huggingface/hub/llama-2-13b/ggml-model-f16.gguf"
HOSTS_PATH="./hosts"
PROMPT_PATH="./prompts/mtbench_prompts.json"

ids=()
prompts=()
while IFS= read -r item; do
    id=$(echo "$item" | jq -r '.[0]')
    prompt=$(echo "$item" | jq -r '.[1]')
    ids+=("$id")
    prompts+=("$prompt")
done < <(jq -c '.[]' $PROMPT_PATH)

n_predicts=(128)
drafts=(4)
n_parrallels=(8)

for n_predict in "${n_predicts[@]}"; do
    echo "Running with n_predict: $n_predict"

    for draft in "${drafts[@]}"; do
        echo "Running with draft: $draft"

        for n_parrallel in "${n_parrallels[@]}"; do
            echo "Running with n_parrallel: $n_parrallel"

            # 遍历数组并执行 mpirun 命令
            for i in "${!ids[@]}"; do
                id="${ids[$i]}"
                prompt="${prompts[$i]}"

                echo "Running with id: $id"
                echo "Prompt: $prompt"

                mpirun -c $((n_parrallel + 1)) -x NCCL_SOCKET_IFNAME=eno12399np0 -x NCCL_IB_DISABLE=0 -x NCCL_IB_CUDA_SUPPORT=1 -mca btl_tcp_if_include eno12399np0  -hostfile ${HOSTS_PATH} --bind-to none \
                    build/bin/speculative \
                    -md ${DRAFT_MODEL_PATH} \
                    -m ${TARGET_MODEL_PATH}  \
                    -e \
                    -n "${n_predict}" \
                    --mpi-layer-split 0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1/1.0 \
                    --ignore-eos \
                    --temp -1.0 \
                    --repeat-last-n 0 \
                    --draft ${draft} \
                    --threads-batch 32 \
                    --threads 32 \
                    --ctx-size 1024 \
                    --p-accept 0 \
                    --p-split 0.001 \
                    --p-recovery 0.4 \
                    --p-decay 0.01 \
                    --batch-size 256 \
                    --cont-batching \
                    --parallel ${n_parrallel} \
                    --result-path ./${n_predict}_${n_parrallel}_${draft}_result.csv \
                    --prompt "${prompt}" >> ./result.log 
                break    
            done        
        done
    done    
done