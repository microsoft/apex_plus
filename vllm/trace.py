from openai import AsyncOpenAI
from openai import OpenAI
import argparse
import requests
import asyncio
import time
import json
import re


APIKEY = "apex123"
BASE_URL = "http://localhost:8000/v1"
client = AsyncOpenAI(api_key=APIKEY, base_url=BASE_URL)
metric_client = OpenAI(api_key=APIKEY, base_url='http://localhost:8000/metrics')
global MODEL

NS_TO_SEC = 1e-9


class Request:
    def __init__(self, req_id: str, prompt: str, prompt_len: int, output_len: int, time_offset: int,total_tokens: int) -> None:
        self.req_id = req_id
        self.prompt = prompt
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.time_offset = time_offset
        self.total_tokens = total_tokens

def parse_trace(trace_file):
    requests: Dict[float, List[Request]] = {}
    with open(trace_file, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            time_offset = json_obj['StartTimeOffset'] * NS_TO_SEC
            req_id = json_obj['RequestID']
            prompt_len = json_obj['ContextTokens']
            output_len = json_obj['GeneratedTokens']
            total_tokens = json_obj['TotalTokens']
            # body = json.loads(json_obj['SynthesizedBody'])
            # body_messages = body['messages']
            body_messages = json_obj['SynthesizedBody']
            # prompt = re.search(r"{{MARK}}(.*?)', 'role'", str(body_messages))
            # parsed_prompt = prompt.group(1).strip()
            parsed_prompt = str(body_messages)
            request = Request(req_id=req_id, prompt=parsed_prompt, prompt_len=prompt_len, output_len=output_len, time_offset=json_obj['StartTimeOffset'] ,total_tokens=total_tokens)
            if time_offset not in requests:
                requests[time_offset] = []
            requests[time_offset].append(request)
    return dict(sorted(requests.items()))


def get_metrics():
    completion = metric_client.completions.create(model=MODEL,
                                      prompt="")
    return completion

def parse_metrics(metrics_text):
    metrics = {}
    for line in metrics_text.splitlines():
        if line and not line.startswith('#'):
            key, value = line.split(' ')
            # metrics[key] = str(value)
            metrics[key] = float(value)
    return metrics

def log_metrics(metric_file):
    metrics = get_metrics()
    # Write to file
    metric_file = open(f'./v06_results/{metric_file}','w')
    parsed_metrics = parse_metrics(metrics)

    for key, value in parsed_metrics.items():        
        line = str(key) + ": " + str(value) + "\n" 
        metric_file.write(line)

     # Calculate TPOT and TTFT
    avg_gen_throuput = 0
    avg_prompt_throughput = 0
    num_gen_tokens = 0
    latency = 0
    tpot_sum = 0
    ttft_sum = 0
    ttft_count = 0
    num_req = 0

    for key, value in parsed_metrics.items():      
        if "avg_generation_throughput" in str(key):
            avg_gen_throuput = value
        elif "avg_prompt_throughput" in str(key):
            avg_prompt_throughput = value
        elif "generation_tokens_total" in str(key):
            num_gen_tokens = value
        elif "request_latency_seconds_sum" in str(key):
            latency = value
        elif "time_per_output_token_seconds_sum" in str(key):
            tpot_sum = value
        elif "time_to_first_token_seconds_sum" in str(key):
            ttft_sum = value
        elif "time_to_first_token_seconds_count" in str(key):
            ttft_count = value
        elif "request_latency_seconds_count" in str(key):
            num_req = value
    
    ttft = ttft_sum/ttft_count
    tpot = tpot_sum/num_gen_tokens

    return ttft,tpot,latency

def validation(responses, validation_file, trace_file, vllm_to_apex_file):
    request_validation: Dict[str,(int,int)] = {}

    # Clear the completion file and keep counters
    prompt_len_trace_sum = 0
    output_len_trace_sum = 0
    prompt_len_vllm_sum = 0
    output_len_vllm_sum = 0
    # for tup in responses:

    reqs = []

    for response,request in responses:
        prompt_len_diff = response.usage.prompt_tokens - request.prompt_len
        output_len_diff = response.usage.completion_tokens - request.output_len
        request_validation[request.req_id] = (prompt_len_diff,output_len_diff)
        completion_file = open(f'./v06_results/{validation_file}','a')
        line = (f'Request: {request.req_id}, prompt difference = {prompt_len_diff}, output_len_diff = {output_len_diff}\n')
        completion_file.write(line)

        # Update counters

        prompt_len_trace_sum += request.prompt_len
        output_len_trace_sum += request.output_len

        prompt_len_vllm_sum += response.usage.prompt_tokens
        output_len_vllm_sum += response.usage.completion_tokens

        # Write a json object to trace file that APEX+ can use 
        # total_compl_tokens = response.usage.completion_tokens + response.usage.completion_tokens
        reqs.append(
            {"StartTime":"2023-10-13T21:23:14.969068Z", 
             "StartTimeOffset":  request.time_offset, 
             "RequestID": request.req_id, 
             "ContextTokens" : response.usage.prompt_tokens,
             "GeneratedTokens" : response.usage.completion_tokens,
             "TotalTokens" : response.usage.prompt_tokens + response.usage.completion_tokens,
            }
        )
       

    # Write to a trace file for apex to use
    # trace_file_base = trace_file.split('.txt')[0]
    # pattern = r"../traces/(.*)"
    # trace_file_base = re.search(pattern, trace_file_base).group(1)

    with open(f'../traces/vllm_to_apex_traces/{vllm_to_apex_file}', 'w') as file:
        for item in reqs:
            file.write(json.dumps(item) + '\n')

    trace_prompt_line = (f'Trace file prompt length sum: {prompt_len_trace_sum}\n')
    trace_output_line = (f'Trace file output length sum: {output_len_trace_sum}\n')

    vllm_prompt_line = (f'vLLm prompt length sum: {prompt_len_vllm_sum}\n')
    vllm_output_line = (f'vLLm output len sum: {output_len_vllm_sum}\n')

    prompt_diff_line = (f'Total prompt len difference(vLLm - trace): {prompt_len_vllm_sum - prompt_len_trace_sum}\n')
    output_len_diff_sum = (f'Total output len difference(vLLm - trace): {output_len_vllm_sum - output_len_trace_sum }\n')

    completion_file.write(trace_prompt_line)
    completion_file.write(trace_output_line)

    completion_file.write(vllm_prompt_line)
    completion_file.write(vllm_output_line)

    completion_file.write(prompt_diff_line)
    completion_file.write(output_len_diff_sum)



async def get_completions(request):
    print(f"Starting request with ID {request.req_id}")
    completion = await client.completions.create(
        model=MODEL,
        prompt=request.prompt,
        max_tokens=request.output_len
    )
    print(f"Completed request with ID {request.req_id}")
    return completion, request


async def send_requests_after_delay(delay, requests, validation_file):
    await asyncio.sleep(delay)
    tasks = [get_completions(request) for request in requests]
    responses = await asyncio.gather(*tasks)

    return responses

    
async def main(requests, validation_file, trace_file, vllm_to_apex_file):
    tasks = [send_requests_after_delay(delay, reqs, validation_file) for delay, reqs in requests.items()]
    responses = await asyncio.gather(*tasks)


    flat_responses = [tup for sublist in responses for tup in sublist]

    validation(flat_responses, validation_file, trace_file, vllm_to_apex_file)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", type=str, default="../traces/irregular_trace.jsonl")
    parser.add_argument("--metric-file", type=str, default="./metrics.txt")
    parser.add_argument("--validation-file", type=str, default="./validation.txt")
    parser.add_argument("--vllm-to-apex-file", type=str)
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-70B")
    args = parser.parse_args()

    MODEL = args.model
    # Clear file and specify trace
    completion_file = open(f'./v06_results/{args.validation_file}','w')
    trace_file_line = (f'Trace file: {args.trace_file}\n')
    completion_file.write(trace_file_line)

    # Parse trace, send requests, create validation file
    requests = parse_trace(args.trace_file)
    asyncio.run(main(requests, args.validation_file, args.trace_file, args.vllm_to_apex_file))
    
    # Log metrics to designated file
    avg_ttft, avg_tpot, avg_latency = log_metrics(args.metric_file)

    # Write Validation file
    completion_file = open(f'./v06_results/{args.validation_file}','a')
    ttft_line = (f'vLLM TTFT: {avg_ttft} \n')
    tpot_line = (f'vLLM TPOT: {avg_tpot} \n')
    latency_line = (f'vLLM Total Latency: {avg_latency} \n')
    completion_file.write(ttft_line)
    completion_file.write(tpot_line)
    completion_file.write(latency_line)