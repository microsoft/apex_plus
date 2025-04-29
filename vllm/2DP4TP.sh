# Bash scipt to recreate experiments and generate graphs
dp_size="2"
tp_size="4" 
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

docker run -d -it --runtime nvidia --gpus '"device=0,1,2,3"' \
        --name "vllm_2DP4TP_0"\
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --env "HUGGING_FACE_HUB_TOKEN=$hf_token"  \
        -v ~/apex:/work/APEX  \
        -p 8000:8000  \
        --ipc=host  vllm/vllm-openai:v0.6.0 \
        --model $full_model\
        --disable-frontend-multiprocessing \
        --gpu-memory-utilization 0.97 \
        --api-key apex123 \
        --tensor-parallel-size $tp_size &\
 docker run -d -it --runtime nvidia --gpus '"device=4,5,6,7"' \
        --name "vllm_2DP4TP_1"\
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --env "HUGGING_FACE_HUB_TOKEN=$hf_token"  \
        -v ~/apex:/work/APEX  \
        -p 8001:8000  \
        --ipc=host  vllm/vllm-openai:v0.6.0 \
        --model $full_model\
        --disable-frontend-multiprocessing \
        --gpu-memory-utilization 0.97 \
        --api-key apex123 \
        --tensor-parallel-size $tp_size &\

# Wait for the server to setup ~ 4 mins
echo "Waiting for server setup"
sleep 260
echo "Running Trace"

# Run the trace script inside the docker container
docker exec -w /work/APEX/vllm vllm_2DP4TP_0 python3 trace.py\
            --model $full_model \
            --trace-file ../traces/${model_name}/${trace_file_base}_2DP4TP_0.jsonl\
            --metric-file ${model_name}_${trace_file_base}_metrics_2DP4TP_0.txt \
            --validation-file ${model_name}_${trace_file_base}_validation_2DP4TP_0.txt \
            --vllm-to-apex-file vllm_${model_name}_${trace_file_base}_2DP4TP_0.jsonl &\
docker exec -w /work/APEX/vllm vllm_2DP4TP_1 python3 trace.py\
            --model $full_model \
            --trace-file ../traces/${model_name}/${trace_file_base}_2DP4TP_1.jsonl\
            --metric-file ${model_name}_${trace_file_base}_metrics_2DP4TP_1.txt \
            --validation-file ${model_name}_${trace_file_base}_validation_2DP4TP_1.txt \
            --vllm-to-apex-file vllm_${model_name}_${trace_file_base}_2DP4TP_1.jsonl &\
# echo $num
wait
sleep 60
echo "Trace finished"

# Clean up
docker stop vllm_2DP4TP_0
docker stop vllm_2DP4TP_1

docker rm vllm_2DP4TP_0
docker rm vllm_2DP4TP_1
echo "Cleaned Up"


# Add file mappings to csv
for i in $(seq 1 $dp_size); do
    echo -e "$experiment_group,${trace_file_base}_2DP4TP_${i}.jsonl,$full_model,1,$num,$num,${model_name}_${trace_file_base}_metrics_2DP4TP_${i}.txt,${model_name}_${trace_file_base}_validation_2DP4TP_${i}.txt,vllm_${model_name}_${trace_file_base}_2DP4TP_${i}.jsonl" >> ./data_mappings_2DP4TP_${i}.csv
done
echo "Saved Data"



python3 apex_run_and_graph.py