# Bash scipt to recreate experiments and generate graphs
tp_size="8" 
dp_size="1"
full_model=$1
model_name=$2
trace_file_base=$3
hf_token=$4


experiment_group=$(tail -n 1 "data_mappings.csv" | awk -F, '{print $1}')

((++experiment_group))

echo "New experiment group: $experiment_group"


########################################################
# vLLM Experiments for 100 Dynamice Irregular Requests #
########################################################

docker run  -d -it --runtime nvidia --gpus all \
        --name "vllm_8TP"\
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --env "HUGGING_FACE_HUB_TOKEN=$hf_token"  \
        -v ~/apex:/work/APEX  \
        -p 8000:8000  \
        --ipc=host  vllm/vllm-openai:v0.6.0 \
        --model $full_model\
        --disable-frontend-multiprocessing \
        --gpu-memory-utilization 0.97 \
        --api-key apex123 \
        --tensor-parallel-size $tp_size \


# Wait for the server to setup ~ 4 mins
echo "Waiting for server setup"
sleep 260
echo "Running Trace"

# Run the trace script inside the docker container
docker exec -w /work/APEX/vllm vllm_8TP python3 trace.py \
            --model $full_model \
            --trace-file ../traces/${model_name}/${trace_file_base}.jsonl\
            --metric-file ${model_name}_${trace_file_base}_metrics_8TP.txt \
            --validation-file ${model_name}_${trace_file_base}_validation_8TP.txt \
            --vllm-to-apex-file vllm_${model_name}_${trace_file_base}_8TP.jsonl \
# echo $num
sleep 60
echo "Trace finished"

# Clean up
docker stop vllm_8TP

docker rm vllm_8TP
echo "Cleaned Up"


# Add file mappings to csv
echo -e "$experiment_group,${trace_file_base}.jsonl,$full_model,1,$num,$num,${model_name}_${trace_file_base}_metrics_8TP.txt,${model_name}_${trace_file_base}_validation_8TP.txt,vllm_${model_name}_${trace_file_base}_8TP.jsonl" >> ./data_mappings_8TP.csv

echo "Saved Data"



python3 apex_run_and_graph.py