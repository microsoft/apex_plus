# Bash scipt to recreate experiments and generate graphs
dp_size="4"
tp_size="2" 
full_model=$1
model_name=$2
trace_file_base=$3
hf_token=$4


experiment_group=$(tail -n 1 "data_mappings.csv" | awk -F, '{print $1}')

((++experiment_group))

echo "New experiment group: $experiment_group"

       
max_model_len_flag="--max-model-len 20560"

/bin/bash mistral_4DP2TP_fp8_dial_2.sh $1 $2 $3 $4 &\
 docker run -d -it --runtime nvidia --gpus '"device=0,1"' \
        --name "vllm_4DP2TP_fp8_0"\
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --env "HUGGING_FACE_HUB_TOKEN=$hf_token"  \
        -v ~/apex:/work/APEX  \
        -p 8000:8000  \
        --ipc=host  vllm/vllm-openai:v0.6.0 \
        --model $full_model\
        --disable-frontend-multiprocessing \
        --gpu-memory-utilization 0.85 \
        --tokenizer_mode mistral \
        --kv-cache-dtype fp8\
        --tokenizer mistralai/Mistral-Large-Instruct-2407\
        --api-key apex123 \
        --tensor-parallel-size $tp_size \
        --disable-custom-all-reduce $max_model_len_flag &\ 
 docker run -d -it --runtime nvidia --gpus '"device=2,3"' \
        --name "vllm_4DP2TP_fp8_1"\
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --env "HUGGING_FACE_HUB_TOKEN=$hf_token"  \
        -v ~/apex:/work/APEX  \
        -p 8001:8000  \
        --ipc=host  vllm/vllm-openai:v0.6.0 \
        --model $full_model\
        --disable-frontend-multiprocessing \
        --gpu-memory-utilization 0.85 \
        --tokenizer_mode mistral \
        --kv-cache-dtype fp8\
        --tokenizer mistralai/Mistral-Large-Instruct-2407\
        --api-key apex123 \
        --tensor-parallel-size $tp_size \
        --disable-custom-all-reduce $max_model_len_flag &\
docker run -d -it --runtime nvidia --gpus '"device=4,5"' \
        --name "vllm_4DP2TP_fp8_2"\
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --env "HUGGING_FACE_HUB_TOKEN=$hf_token"  \
        -v ~/apex:/work/APEX  \
        -p 8002:8000  \
        --ipc=host  vllm/vllm-openai:v0.6.0 \
        --model $full_model\
        --disable-frontend-multiprocessing \
        --gpu-memory-utilization 0.85 \
        --tokenizer_mode mistral \
        --kv-cache-dtype fp8\
        --tokenizer mistralai/Mistral-Large-Instruct-2407\
        --api-key apex123 \
        --tensor-parallel-size $tp_size \
        --disable-custom-all-reduce $max_model_len_flag &\ 
 docker run -d -it --runtime nvidia --gpus '"device=6,7"' \
        --name "vllm_4DP2TP_fp8_3"\
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --env "HUGGING_FACE_HUB_TOKEN=$hf_token"  \
        -v ~/apex:/work/APEX  \
        -p 8003:8000  \
        --ipc=host  vllm/vllm-openai:v0.6.0 \
        --model $full_model\
        --disable-frontend-multiprocessing \
        --gpu-memory-utilization 0.85 \
        --tokenizer_mode mistral \
        --kv-cache-dtype fp8\
        --tokenizer mistralai/Mistral-Large-Instruct-2407\
        --api-key apex123 \
        --tensor-parallel-size $tp_size \
        --disable-custom-all-reduce $max_model_len_flag &\