# vLLM Docker Image Setup
For experiments, the official Docker image for deployment vLLM provides was used. The Docker image is available on Docker Hub and can be pulled with the command:

```bash
docker pull vllm/vllm-openai:latest
```

# A Note on Running Experiments 
To run experiments, the flow can be summarized to:

1. Start a deployment server by running a similar docker command shown in the Running vLLM section. 
2. Once the server is setup, attach a shell or visual studio code to the docker container and navigate to the /APEX/vllm directory
3. Send requests to the vllm server by providing trace.py with a trace file and names for raw metric and validation files


# Running vLLM
Using the provided docker image will create a server instance through an entrypoint. This means vLLM engine arguments can be passed in when running a container. A list of arguments can be found at https://docs.vllm.ai/en/stable/models/engine_args.html. A sample command is:
```bash
docker run -it --runtime nvidia --gpus all   -v ~/.cache/huggingface:/root/.cache/huggingface   --env "HUGGING
_FACE_HUB_TOKEN=<HF_TOKEN>"  -v ~/APEX:/work/APEX  -p 8000:8000   --ipc=host   vllm/vllm-openai:latest --model meta-llama/Meta-Llama-3.1-70B --disable-frontend-multiprocessing --gpu-memory-utilization 0.99 --api-key apex123 --tensor-parallel-size 8 --disable-custom-all-reduce  <more args>
```


Sending requests can be done via the trace.py script in the /vllm directory. This can be through a command in the form of:

```bash
python3 trace.py --trace-file <trace_file> --metric-file <new_metric_file_name> --validation-file <new_validation_file_name>
```
Trace files are searched for in the /traces directory of APEX and the trace.py script requires trace objects with the fields:
1. StartTimeOffset
2. RequestID
3. ContextTokens
4. GeneratedTokens

since these are parsed. Likewise, the metric-file and validation-file flags are used to specify names for files the scripts generates. 

The metric file will store raw values from the server /metrics endpoint, as specified at https://docs.vllm.ai/en/latest/serving/metrics.html. If not provided, this will default to metrics.txt.

The validation file will store some parsed fields from the /metrics endpoint like TPOT, TTFT, etc. but also provide data for prompt and output token differences between the trace file and the completed request. If not provided, this will default to validation.txt.


# Reproducing the experiments
To reproduce the experimental results in the paper, use the command
```bash
bash llama.sh
bash mistral.sh
```
Note that you would need to insert your Huggingface token in the first line of the above bash scripts.