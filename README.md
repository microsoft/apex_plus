# APEX+

APEX+ is an extensible and dynamism-aware simulator for **A**utomated **P**arallel **EX**ecution in LLM serving.
Given an LLM, a device cluster, and input requests with varying context/generation lengths, APEX+ generates an optimal parallel execution plan for LLM serving.
APEX+ performs dynamism-aware simulation to model iteration-level batching, and leverages LLMs' repetitive structure to reduce design space, scaling efficiently to trillion-scale models. 
APEX+ finds plans up to 3.37x faster than heuristics, and also plans that reduce energy consumption by up to 45% compared to latency-optimal plans.
APEX+ performs comprehensive evaluations, reporting key system metrics like time per output token (TPOT) and time to first token (TTFT), which can help service providers meet SLOs.
APEX+ identifies an optimal plan within 15 minutes on a CPU, making it 71x faster and 1234x more cost-effective than cloud-based GPU deployment.

Currently, APEX+ includes op-level profiling data for the following backends: V100-16GB, H100 SXM 80GB, and H200 SXM 141 GB.
To experiment on different cluster backends, one need to create a folder of the backend name (e.g., H100-SXM-80GB) and store the profiling files in `profile/comp/{Backend}` and `profile/comm/{Backend}`. Note that APEX+ also supports hardware backends other than GPUs, as long as their op-level profiling data is provided. Detailed instructions to obtain the profiling data are in the [README](./profile/scripts/README.md) under the profile folder.


## Environment Setup
### Installing dependencies

Install the dependencies by running:
```bash
pip install -r requirements.txt
```

## Running APEX+

Run `main.py` in the root directory of the repository. We can simulate various models:
```bash
# Simulate the decoder-only model llama3-70b; the ``--all`` flag prints the simulation results of all the execution plans, otherwise only the latency-optimal plan is printed
python main.py --model llama3-70b --num-gpus-per-node 2 --prompt-len 128 --output-len 2048 --all

# Simulate the encoder-decoder model Whisper
python main.py --model whisper --num-gpus-per-node 2 --prompt-len 128 --output-len 2048

# Simulate with a trace file
python main.py --model llama3-70b --num-gpus-per-node 2 --trace-file ./traces/llama/creation_05.jsonl

# Simulate with quantization mode W8A16 and KV-cache in FP8
python main.py --model llama3-70b --num-gpus-per-node 2 --trace-file ./traces/llama/creation_05.jsonl --kv-dtype float8 --weight-dtype float8 --activation-dtype half

# Simulate with a MoE Model
python main.py --model mixtral-8x7b-local --num-gpus-per-node 4 --num-experts 8 --trace-file ./traces/mistral/lmsys_05.jsonl
```
> Note: APEX+ supports simulation on real request traces. The traces should be stored in `.jsonl` format, and include the following items: StartTimeOffset (offset from the first request, ns), ContextTokens (input sequence length), and GeneratedTokens (output sequence length).  
> Note 2: For a full list of supported LLMs, please see ``main.py``.

## Validating APEX+ with vLLM and SGLang

To compare the simulation results with actual LLM serving behavior, we use vLLM to serve dense LLMs, and SGLang to serve MoE models. The instructions to setup and run vLLM/SGLang experiments are in the `README.md` files of their respective folders. We also include the experimental results of serving several LLMs on a cluster with eight H100-SXM-80GB GPUs in a folder named _results_.

## Output example of APEX+
Using Llama3-70b as an example, the output will look like:
```
Namespace(model='./apex_plus/models/llama3_70b_config.json', num_experts=None, topk=2, capacity_factor=1.0, num_nodes=1, num_gpus_per_node=2, gpu='H100-SXM-80GB', frequency=1980, trace_file='./traces/simplified_irregular_trace.jsonl', prompt_len=2048, output_len=128, num_requests=1024, disable_ray=False, kv_dtype='half', weight_dtype='half', activation_dtype='half', all=True, request_percentiles=[], token_percentiles=[])
<bound method LLaMA3.__repr__ of LLaMA3(vocab_size=128256, num_layers=80, num_heads=64, num_kv_heads=8, hidden_size=8192, intermediate_size=28672)>
Generated 3 decoder candidate plans.
================================================================================
* Parallel schedule 0 for decoder:
  # Model replicas: 1
  # Stages: 1
  # Blocks per stage: 80
--------------------------------------------------------------------------------
  Name        Value
--------------------------------------------------------------------------------
  MQA         1 replicas, TaskMapping(tasks_per_device={'MQAHead': 32})x2
  AllReduce   2 devices
  SwiGLU      1 replicas, TaskMapping(tasks_per_device={'SwiGLUFilter':
              14336})x2
  AllReduce   2 devices
  AllGather   2 devices
--------------------------------------------------------------------------------
* Statistics:
--------------------------------------------------------
  Name                                           Value
--------------------------------------------------------
  Parameter size per device (GB)                 63.8
  Activation memory per device (GB)              16.2
  Avg. requests per iteration (per microbatch)   14.4
  Avg. tokens per iteration (per microbatch)     96.8
--------------------------------------------------------
* Performance Metrics:
--------------------------------------------------------------------------------
  Name                                                 Value      Units
--------------------------------------------------------------------------------
  Throughput: Avg. Tokens generated per second         336.752    tokens/sec
  Throughput: Avg. Tokens processed per second         2266.710   tokens/sec
  Throughput: Requests per second                      1.611      requests/sec
  Latency: Avg. Time to first token (TTFT in msec)     200.707    msec
  Latency: Avg. Time per output token (TPOT in msec)   44.064     msec
  Avg. TBT Percentile: P50                             42.578     msec
  Avg. TBT Percentile: P95                             54.281     msec
  Request Completion Latency: 50th percentile          7.749      sec
  Request Completion Latency: 95th percentile          20.692     sec
  Avg. MFU Per iteration                               6.591      %
  MBU                                                  43.646     %
--------------------------------------------------------------------------------
* Time breakdown:
---------------------------------------
  Name        Time (sec)    Ratio (%)
---------------------------------------
  MQA         17.23 ± 0.0   27.8
  AllReduce   3.56 ± 0.0    5.7
  SwiGLU      39.43 ± 0.0   63.5
  AllGather   1.87 ± 0.0    3.0
  Total       62.06         100.0
---------------------------------------
Energy Consumption: 75.69 KJ
================================================================================
* Parallel schedule 1 for decoder:
  # Model replicas: 1
  # Stages: 1
  # Blocks per stage: 80
------------------------------------------------------------------------------------
  Name            Value
------------------------------------------------------------------------------------
  MQA             2 replicas, TaskMapping(tasks_per_device={'MQAHead': 64})x1
  ReduceScatter   1 devices
  AllGather       2 devices
  SwiGLU          1 replicas, TaskMapping(tasks_per_device={'SwiGLUFilter':
                  14336})x2
  AllReduce       2 devices
  AllToAll        2 devices
  AllGather       1 devices
------------------------------------------------------------------------------------
* Statistics:
--------------------------------------------------------
  Name                                           Value
--------------------------------------------------------
  Parameter size per device (GB)                 75.0
  Activation memory per device (GB)              5.0
  Avg. requests per iteration (per microbatch)   15.4
  Avg. tokens per iteration (per microbatch)     104.1
--------------------------------------------------------
* Performance Metrics:
--------------------------------------------------------------------------------
  Name                                                 Value      Units
--------------------------------------------------------------------------------
  Throughput: Avg. Tokens generated per second         243.242    tokens/sec
  Throughput: Avg. Tokens processed per second         1637.286   tokens/sec
  Throughput: Requests per second                      1.164      requests/sec
  Latency: Avg. Time to first token (TTFT in msec)     348.100    msec
  Latency: Avg. Time per output token (TPOT in msec)   50.885     msec
  Avg. TBT Percentile: P50                             50.873     msec
  Avg. TBT Percentile: P95                             55.442     msec
  Request Completion Latency: 50th percentile          9.138      sec
  Request Completion Latency: 95th percentile          19.982     sec
  Avg. MFU Per iteration                               4.761      %
  MBU                                                  37.795     %
--------------------------------------------------------------------------------
* Time breakdown:
---------------------------------------
  Name        Time (sec)    Ratio (%)
---------------------------------------
  MQA         25.80 ± 1.9   30.0
  AllGather   1.32 ± 0.1    1.5
  SwiGLU      38.05 ± 2.6   44.3
  AllReduce   1.67 ± 0.1    1.9
  AllToAll    1.68 ± 0.1    2.0
  Idle        17.39 ± 4.8   20.2
  Total       85.92         100.0
---------------------------------------
Energy Consumption: 85.91 KJ
================================================================================
* Parallel schedule 2 for decoder:
  # Model replicas: 1
  # Stages: 2
  # Blocks per stage: 40
--------------------------------------------------------------------------------
  Name        Value
--------------------------------------------------------------------------------
  MQA         1 replicas, TaskMapping(tasks_per_device={'MQAHead': 64})x1
  AllReduce   1 devices
  SwiGLU      1 replicas, TaskMapping(tasks_per_device={'SwiGLUFilter':
              28672})x1
  AllReduce   1 devices
  AllGather   1 devices
--------------------------------------------------------------------------------
* Statistics:
--------------------------------------------------------
  Name                                           Value
--------------------------------------------------------
  Parameter size per device (GB)                 63.8
  Activation memory per device (GB)              16.2
  Avg. requests per iteration (per microbatch)   5.6
  Avg. tokens per iteration (per microbatch)     37.7
--------------------------------------------------------
* Performance Metrics:
--------------------------------------------------------------------------------
  Name                                                 Value      Units
--------------------------------------------------------------------------------
  Throughput: Avg. Tokens generated per second         163.343    tokens/sec
  Throughput: Avg. Tokens processed per second         1099.480   tokens/sec
  Throughput: Requests per second                      0.782      requests/sec
  Latency: Avg. Time to first token (TTFT in msec)     279.278    msec
  Latency: Avg. Time per output token (TPOT in msec)   62.767     msec
  Avg. TBT Percentile: P50                             61.840     msec
  Avg. TBT Percentile: P95                             66.089     msec
  Request Completion Latency: 50th percentile          10.981     sec
  Request Completion Latency: 95th percentile          24.774     sec
  Avg. MFU Per iteration                               3.193      %
  MBU                                                  30.640     %
--------------------------------------------------------------------------------
* Time breakdown:
--------------------------------------
  Name       Time (sec)    Ratio (%)
--------------------------------------
  MQA        34.17 ± 1.0   26.7
  SwiGLU     83.28 ± 2.6   65.1
  SendRecv   0.04 ± 0.0    0.0
  Idle       10.46 ± 3.6   8.2
  Total      127.95        100.0
--------------------------------------
Energy Consumption: 158.52 KJ
================================================================================
```
The result indicates that the best execution plan (parallel schedule 0) for this specific case is to use Megatron-style 2-way tensor parallelism for both MQA and SwiGLU layers.
APEX+ also shows that the third best plan (parallel schedule 2) is to use 2-way pipeline parallelism.


Please refer to another [README](./profile/scripts/README.md) for instructions on profiling.

## Setting SLOs and Max Batch Size 

APEX is capable of providing insights into the percentage of requests that meet a target SLO. By deafult, APEX enables SLOs to default values of 10 msec for both TTFT and TPOT.

A max batch size can also be set for APEX. Both functionalities are configured with the 3 corresponding flags shown below
```bash
python main.py --num-gpus 8 --num-nodes 1 --model llama3-70b --num-requests 10 --ttft-slo 100 --tpot-slo 30 --max-batch 10
```
The APEX output will include a SLO Metrics based off the candidate plan.
```bash
--------------------------------------------------------------------------------
* Latency SLO Metrics:
* Batch Size: 10
-----------------------------------------
  Name   Target SLO(msec)   SLOs Met(%)  
-----------------------------------------
  TTFT   100                0.000 %      
  TPOT   30                 100.000 %    
-----------------------------------------
```
## Our paper
The paper of this work can be accessed [here](https://arxiv.org/abs/2411.17651).  
DOI: 10.5281/zenodo.15300595
```
@misc{apex,
      title={Toward High-Performance LLM Serving: A Simulation-Based Approach for Identifying Optimal Parallelism}, 
      author={Yi-Chien Lin and Woosuk Kwon and Ronald Pineda and Fanny Nina Paravecino},
      year={2024},
      eprint={2411.17651},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2411.17651}, 
}
```
