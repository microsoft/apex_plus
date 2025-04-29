# Bash scipt to recreate experiments and generate graphs

tp_size=("8")
hf_token=$1
trace_file_base="fp8_summarization_dynamic_trace"
trace_file="summarization_dynamic_trace.jsonl"
model="Meta-Llama-3.1-70B-Instruct-FP8"
full_model="neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8"


experiment_group=$(tail -n 1 "data_mappings.csv" | awk -F, '{print $1}')

((++experiment_group))

echo "New experiment group: $experiment_group"


for num in "${tp_size[@]}"; do

    max_model_len_flag=""

    # If tp = 2, prevent KV cache error by running different command
    if [ "$num" -eq 2 ]; then
        echo "Num GPUs set to 2, adding extra flag..."
        max_model_len_flag="--max-model-len 61360"
    fi


    docker run -d -it --runtime nvidia --gpus all \
            --name "vllm_tp_$num"\
            -v ~/.cache/huggingface:/root/.cache/huggingface \
            --env "HUGGING_FACE_HUB_TOKEN=$hf_token"  \
            -v ~/APEX:/work/APEX  \
            -p 8000:8000  \
            --ipc=host  vllm/vllm-openai:latest \
            --model neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8\
            --disable-frontend-multiprocessing \
            --gpu-memory-utilization 0.99 \
            --api-key apex123 \
            --tensor-parallel-size $num \
            --disable-custom-all-reduce $max_model_len_flag\
            --kv-cache-dtype fp8\


    # Wait for the server to setup ~ 4 mins
    echo "Waiting for server setup"
    sleep 260
    echo "Running Trace"

    # Run the trace script inside the docker container
    docker exec -w /work/APEX/vllm vllm_tp_$num python3 trace_fp8.py \
                --trace-file ../traces/$trace_file\
                --metric-file ${model}_tp_${num}_${trace_file_base}_metrics.txt \
                --validation-file ${model}_tp_${num}_${trace_file_base}_validation.txt \
                --vllm-to-apex-file vllm_${model}_tp_${num}_${trace_file_base}.jsonl \
    # echo $num
    sleep 60
    echo "Trace finished"

    # Clean up
    docker stop vllm_tp_$num

    docker rm vllm_tp_$num
    echo "Cleaned Up"


    # Add file mappings to csv
    echo -e "$experiment_group,$trace_file,$full_model,1,$num,$num,${model}_tp_${num}_${trace_file_base}_metrics.txt,${model}_tp_${num}_${trace_file_base}_validation.txt,vllm_${model}_tp_${num}_${trace_file_base}.jsonl" >> ./data_mappings.csv

    echo "Saved Data"

done

python3 apex_run_and_graph.py