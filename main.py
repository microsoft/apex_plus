import argparse

from apex_plus.cluster.cluster import Cluster
from apex_plus.models.registry import get_model_ir
from apex_plus.search.engine import SearchEngine
from apex_plus.simulator.trace import Trace
from apex_plus.utils.dtype import DTYPE, _DTYPE_REGISTRY

# NOTE: This is not a complete list of supported models.
SHORTCUT = {
    "bloom": "bigscience/bloom",
    "llama-7b": "huggyllama/llama-7b",
    "llama-13b": "huggyllama/llama-13b",
    "llama-30b": "huggyllama/llama-30b",
    "llama-65b": "huggyllama/llama-65b",
    "gpt-j": "EleutherAI/gpt-j-6b",
    "gpt-neox": "EleutherAI/gpt-neox-20b",
    "wizardcoder": "WizardLM/WizardCoder-15B-V1.0",
    "whisper": "openai/whisper-large-v3",
    "clip": "openai/clip-vit-large-patch14",
    # HF authentication token is needed if there is local config.json file
    # Configuration file can be taken from HF model repo
    "mistral-7b-local": "./apex_plus/models/mistral_config.json",
    "mistral-7b": "teknium/OpenHermes-2.5-Mistral-7B",
    "t5": "google/flan-t5-xxl",
    "llama3-70b": "./apex_plus/models/llama3_70b_config.json",
    "mixtral-8x7b-local": "./apex_plus/models/mixtral8x7b_config.json",
    "mixtral-8x7b": "cognitivecomputations/dolphin-2.5-mixtral-8x7b",
    "mixtral-8x22b-local": "./apex_plus/models/mixtral8x22b_config.json",
    "mixtral-8x22b": "mistral-community/Mixtral-8x22B-v0.1-AWQ",
    "llama3.1-70b": "./apex_plus/models/llama3.1_70b_config.json",
    "llama3.1-8b": "./apex_plus/models/llama3.1_8b_config.json",
}


def get_model_shortcuts():
    return SHORTCUT


def main(args: argparse.Namespace):
    if args.model in SHORTCUT:
        args.model = SHORTCUT[args.model]
    print(args)

    model, model_config = get_model_ir(
        args.model, args.num_experts, args.topk, args.capacity_factor
    )

    encoder_cluster = Cluster.from_gpu(args.gpu, args.num_nodes, 1)
    cluster = Cluster.from_gpu(args.gpu, args.num_nodes, args.num_gpus_per_node)

    if args.trace_file:
        trace = Trace.from_dynamic(args.trace_file)
    else:
        trace = Trace.from_static(args.num_requests, args.prompt_len, args.output_len)

    dtype = {
        "kv": _DTYPE_REGISTRY[args.kv_dtype],
        "w": _DTYPE_REGISTRY[args.weight_dtype],
        "act": _DTYPE_REGISTRY[args.activation_dtype],
    }

    if model.num_encoder_blocks == 0 and model.num_decoder_blocks == 0:
        raise RuntimeError("Number of encoders and decoders cannot both be zero.")
    if model.num_encoder_blocks > 0:
        engine = SearchEngine(
            model, encoder_cluster, trace, "encoder", dtype
        )  # search for encoder
        _, trace = engine.search(
            args.all,
            args.frequency,
            args.request_percentiles,
            args.token_percentiles,
            model_config, 
            args.ttft_slo, 
            args.tpot_slo, 
            args.max_batch_size,
        )  # updated traces by adding encode time
        trace = Trace(trace)
    if model.num_decoder_blocks > 0:
        engine = SearchEngine(
            model, cluster, trace, "decoder", dtype
        )  # search for decoder
        _, trace = engine.search(
            args.all,
            args.frequency,
            args.request_percentiles,
            args.token_percentiles,
            model_config, 
            args.ttft_slo, 
            args.tpot_slo, 
            args.max_batch_size,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Model name in HuggingFace model Hub"
    )
    # MoE config
    parser.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="Number of MLP experts of the model. Default is none for models not regarded as MOE model.",
    )
    parser.add_argument(
        "--topk", type=int, default=2, help="Topk hyperparameter for MOE models."
    )
    parser.add_argument(
        "--capacity-factor",
        type=float,
        default=1.0,
        help="Capacity factor for MoE models",
    )
    # Cluster config
    parser.add_argument(
        "--num-nodes", type=int, default=1, help="Number of nodes in the cluster."
    )
    parser.add_argument(
        "--num-gpus-per-node",
        type=int,
        required=True,
        help=" Number of GPUs per node in the cluster",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        choices=["V100-PCIE-16GB", "H100-SXM-80GB","H200-SXM-141GB",],
        default="H100-SXM-80GB",
    )
    parser.add_argument("--frequency", type=int, choices=[0, 810, 1980], default=0)
    # Workload config
    parser.add_argument("--trace-file", type=str)
    parser.add_argument("--prompt-len", type=int, default=2048)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1024,
        help="Number of requests to feed into the APEX. Large number of requests increase simulation accuracy but also increases latency of simulation execution.",
    )
    # Misc
    parser.add_argument(
        "--disable-ray",
        action="store_true",
        help="Disable Ray and serialize the execution of " "simulation.",
    )
    # Quantization
    parser.add_argument(
        "--kv-dtype", type=str, choices=["float", "half", "float8"], default="half"
    )
    parser.add_argument(
        "--weight-dtype", type=str, choices=["float", "half", "float8"], default="half"
    )
    parser.add_argument(
        "--activation-dtype",
        type=str,
        choices=["float", "half", "float8"],
        default="half",
    )

    # Output config
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Output all possible execution plans. Defaults to False",
    )
    # Log Additional Percentiles
    parser.add_argument(
        "--request-percentiles",
        type=int,
        default=[],
        nargs="+",
        help="Output specified percentiles in addition to P50 and P95 for request latencies",
    )
    parser.add_argument(
        "--token-percentiles",
        type=int,
        default=[],
        nargs="+",
        help="Output specified percentiles in addition to P50 and P95 for token generation latencies",
    )
    # Define SLO in ms
    parser.add_argument(
        "--ttft-slo", 
        type=int, default=10,
        help="Define SLO Latency for TTFT in ms. Default is 10 ms"
    )
    parser.add_argument(
        "--tpot-slo", 
        type=int, 
        default=10,
        help="Define SLO Latency for TPOT in ms. Default is 10 ms"
    )
    # Define max batch size
    parser.add_argument(
        "--max-batch-size", 
        type=int, 
        default=0,
        help="Define max batch size. This is also known as max number of sequences."
        )
    args = parser.parse_args()

    main(args)
