# Bash scipt to recreate experiments and generate graphs
tp_size="2" 
dp_size="1"
full_model=$1
model_name=$2
hf_token=$3

# run on datasets for 2TP

experiment_group=$(tail -n 1 "data_mappings.csv" | awk -F, '{print $1}')

((++experiment_group))

echo "New experiment group: $experiment_group"

max_model_len_flag="--max-model-len 24272"
docker run  -d -it --runtime nvidia --gpus '"device=4,5"' \
        --name "vllm_2TP"\
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --env "HUGGING_FACE_HUB_TOKEN=$hf_token"  \
        -v ~/apex:/work/APEX  \
        -p 8001:8000  \
        --ipc=host  vllm/vllm-openai:v0.6.0 \
        --model $full_model\
        --disable-frontend-multiprocessing \
        --gpu-memory-utilization 0.97 \
        --api-key apex123 \
        --tensor-parallel-size $tp_size \
        --disable-custom-all-reduce $max_model_len_flag\


# Wait for the server to setup ~ 4 mins
echo "Waiting for server setup"
sleep 260
echo "Running Trace"

# Run the trace script inside the docker container
trace_file_base="lmsys_05"
docker exec -w /work/APEX/vllm vllm_2TP python3 trace.py \
            --model $full_model \
            --trace-file ../traces/${model_name}/${trace_file_base}.jsonl\
            --metric-file ${model_name}_${trace_file_base}_metrics_2TP.txt \
            --validation-file ${model_name}_${trace_file_base}_validation_2TP.txt \
            --vllm-to-apex-file vllm_${model_name}_${trace_file_base}_2TP.jsonl \

trace_file_base="creation_05"
docker exec -w /work/APEX/vllm vllm_2TP python3 trace.py \
            --model $full_model \
            --trace-file ../traces/${model_name}/${trace_file_base}.jsonl\
            --metric-file ${model_name}_${trace_file_base}_metrics_2TP.txt \
            --validation-file ${model_name}_${trace_file_base}_validation_2TP.txt \
            --vllm-to-apex-file vllm_${model_name}_${trace_file_base}_2TP.jsonl \

trace_file_base="summarization_05"
docker exec -w /work/APEX/vllm vllm_2TP python3 trace.py \
            --model $full_model \
            --trace-file ../traces/${model_name}/${trace_file_base}.jsonl\
            --metric-file ${model_name}_${trace_file_base}_metrics_2TP.txt \
            --validation-file ${model_name}_${trace_file_base}_validation_2TP.txt \
            --vllm-to-apex-file vllm_${model_name}_${trace_file_base}_2TP.jsonl \
# echo $num
sleep 60
echo "Trace finished"

# Clean up
docker stop vllm_2TP

docker rm vllm_2TP
echo "Cleaned Up"

echo "Saved Data"