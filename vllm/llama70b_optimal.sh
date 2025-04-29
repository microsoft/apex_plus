# Bash scipt to recreate experiments and generate graphs
tp_size="1" 
dp_size="4"
pp_size="2"
hf_token=$1
trace_file_base="summarization_dynamic_trace"
# trace_file="summarization_dynamic_trace.jsonl"
model="Meta-Llama-3.1-70B"
full_model="meta-llama/Meta-Llama-3.1-70B"


experiment_group=$(tail -n 1 "data_mappings.csv" | awk -F, '{print $1}')

((++experiment_group))

echo "New experiment group: $experiment_group"

# Start servers for different tp sizes
for i in $(seq 1 $dp_size); do
    
    max_model_len_flag=""
    if [ "$dp_size" -eq 4 ]; then
        echo "Num GPUs set to 2, adding extra flag..."
        max_model_len_flag="--max-model-len 52720"
    fi

    docker run -d -it --runtime nvidia --gpus all \
           --name "vllm_optimal"\
           -v ~/.cache/huggingface:/root/.cache/huggingface \
           --env "HUGGING_FACE_HUB_TOKEN=$hf_token"  \
           -v ~/apex:/work/APEX  \
           -p 8000:8000  \
           --ipc=host  vllm/vllm-openai:v0.5.4 \
           --model $full_model\
           --disable-frontend-multiprocessing \
           --gpu-memory-utilization 0.97 \
           --api-key apex123 \
           --tensor-parallel-size $tp_size \
           --pipeline-parallel-size $pp_size \
           --disable-custom-all-reduce $max_model_len_flag\


    # Wait for the server to setup ~ 4 mins
    echo "Waiting for server setup"
    sleep 260
    echo "Running Trace"

    # Run the trace script inside the docker container
    docker exec -w /work/APEX/vllm vllm_optimal python3 trace.py \
                --trace-file ../traces/${trace_file_base}_${i}.jsonl\
                --metric-file ${model}_optimal_${trace_file_base}_metrics_${i}.txt \
                --validation-file ${model}_optimal_${trace_file_base}_validation_${i}.txt \
                --vllm-to-apex-file vllm_${model}_optimal_${trace_file_base}_${i}.jsonl \
    # echo $num
    sleep 60
    echo "Trace finished"

    # Clean up
    docker stop vllm_optimal

    docker rm vllm_optimal
    echo "Cleaned Up"


    # Add file mappings to csv
    echo -e "$experiment_group,${trace_file_base}_${i}.jsonl,$full_model,1,$num,$num,${model}_optimal_${trace_file_base}_metrics_${i}.txt,${model}_optimal_${trace_file_base}_validation_${i}.txt,vllm_${model}_optimal_${trace_file_base}_${i}.jsonl" >> ./data_mappings_${i}.csv

    echo "Saved Data"

done

python3 apex_run_and_graph.py