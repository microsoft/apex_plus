# Bash scipt to recreate experiments and generate graphs
tp_size="8" 
full_model=$1
model_name=$2
trace_file_base=$3
hf_token=$4


experiment_group=$(tail -n 1 "data_mappings.csv" | awk -F, '{print $1}')

((++experiment_group))

echo "New experiment group: $experiment_group"


docker run  -d -it --runtime nvidia --gpus all \
        --shm-size 32g \
        --name "sglang_${tp_size}TP"\
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --env "HUGGING_FACE_HUB_TOKEN=$hf_token"  \
        -v ~/work/APEX_github/apex:/work/APEX  \
        -p 8000:8000 \
        --ipc=host  sglang:mixtral \


docker exec -w /work/APEX/sglang sglang_${tp_size}TP bash -c "\
nohup python3 -m sglang.launch_server \
  --model-path mistralai/Mixtral-8x22B-Instruct-v0.1 \
  --host 0.0.0.0 \
  --port 8000 \
  --disable-cuda-graph \
  --attention-backend triton \
  --tp ${tp_size} \
  --enable-metrics > server.log 2>&1 &"

echo "Waiting for sglang server to respond..."
# For some reason server returns a 404 and not 200, but at least we know it is up
until curl -s -o /dev/null -w "%{http_code}" http://localhost:8000 | grep -q "404"; do
  sleep 2
done
# sleep 180
echo "sglang server is up!"

echo "Sending Requests"
docker exec -w /work/APEX/sglang sglang_${tp_size}TP python3 trace.py \
            --trace-file ../traces/${model_name}/${trace_file_base}.jsonl \
            --metric-file ${model_name}_tp_${tp_size}_${trace_file_base}_metrics.txt \
            --validation-file ${model_name}_tp_${tp_size}_${trace_file_base}_validation.txt \
            --sglang-to-apex-file sglang_${model_name}_tp_${tp_size}_${trace_file_base}.jsonl \

echo "Trace finished"
docker stop sglang_${tp_size}TP
docker rm sglang_${tp_size}TP
echo "Cleaned Up"




echo -e "$experiment_group,${trace_file_base}.jsonl,$full_model,1,$num,$num,${model_name}_${trace_file_base}_metrics_${tp_size}TP.txt,${model_name}_${trace_file_base}_validation_${tp_size}TP.txt,sglang_${model_name}_${trace_file_base}_${tp_size}TP.jsonl" >> ./data_mappings_${tp_size}TP.csv
echo "Saved Data"
