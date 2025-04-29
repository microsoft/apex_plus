# Profiling Computation & Communication Overheads

## Setup

Install the dependencies by running:
```bash
pip install -r requirements.txt
```
You should have NVIDIA GPUs and CUDA installed on your machine.
The profiling scripts were tested on CUDA 12.1 and NCCL 2.18.3.

Before running the commands below, set up a Ray cluster by running:
```bash
ray start --head  # On one of the nodes.

ray start --address <public address of head node:6379 (e.g., '10.2.1.5:6379')>  # On the other nodes.
```

Check status of ray
```bash
ray status
```


## GEMM (half and float)

```bash
python gemm.py \
  --gpu <GPU name (e.g., V100-PCIE-16GB)> \
  --num-gpus <total number of GPUs in the Ray cluster> \
  --dtype <data type (e.g., half)>
```

## Multi-head attention

```bash
python mha.py \
  --gpu <GPU name (e.g., V100-PCIE-16GB)> \
  --num-gpus <total number of GPUs in the Ray cluster> \
  --dtype <data type (e.g., half)>
```

## Collective communication

```bash
python comm.py
```
