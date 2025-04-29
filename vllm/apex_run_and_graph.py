import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import csv
import os
import re


def get_vllm_results(metric_file, validation_file):
    # Parse ttft and tpot from metric and validation files
    with open("./results/" + validation_file, 'r') as file:
        lines = file.readlines()
    
    ttft = 0
    tpot = 0

    # Iterate through each line
    for line in lines:
        if "vLLM TTFT:" in line:
            # Split the line at the colon and strip any whitespace
            ttft = line.split(': ')[1].strip()
        elif "vLLM TPOT:" in line:
            tpot = line.split(': ')[1].strip()
    
    ttft_msec = float(ttft) * 1000
    tpot_msec = float(tpot) * 1000
    return ttft_msec, tpot_msec




def run_apex(trace_file,model,num_nodes,num_gpus):
    print(f"Running APEX+ experiment for: {trace_file} for {model} model with {num_nodes} nodes and {num_gpus} gpus")
    os.chdir("./..")

    trace_path = "./traces/vllm_to_apex_traces/" + trace_file

    if model == "meta-llama/Meta-Llama-3.1-70B":
        model = "llama3-70b"


    apex_output = subprocess.check_output(["python3", "main.py", "--model", str(model),"--num-nodes",str(num_nodes), "--num-gpus-per-node",str(num_gpus),
                                               "--trace-file",trace_path], text=True)
    print(apex_output)

    # Parse TPOT and TTFT
    ttft_pattern = re.compile(r"Latency: Avg\. Time to first token \(TTFT in msec\)\s+([\d.]+)\s+msec")
    tpot_pattern = re.compile(r"Latency: Avg\. Time per output token \(TPOT in msec\)\s+([\d.]+)\s+msec")
    ttft_match = ttft_pattern.search(apex_output)
    tpot_match = tpot_pattern.search(apex_output)
    ttft_value = float(ttft_match.group(1))
    tpot_value = float(tpot_match.group(1))
    os.chdir("./vllm")

    return ttft_value, tpot_value

def sub_plot(x_axis, vllm_data, apex_data, data_group, experiment_group):
    plt.plot(x_axis, vllm_data, label="vLLM")
    plt.plot(x_axis, apex_data, label="APEX+")

    plt.title(f"APEX+ vs vLLM: Normalized {data_group} Speedup")
    plt.xlabel("Num GPUs")
    plt.ylabel("Normalized Speedup")
    plt.legend()

    # Add data labels
    for i, (xi, yi) in enumerate(zip(x_axis, vllm_data)):
        plt.annotate(f'({xi}, {yi})', (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center')

    for i, (xi, yi) in enumerate(zip(x_axis, apex_data)):
        plt.annotate(f'({xi}, {yi})', (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.savefig(f"./plots/{data_group}_group_{experiment_group}.jpg")
    plt.clf()

def plot(x_axis, vllm_ttft_data, vllm_tpot_data, apex_ttft_data, apex_tpot_data, experiment_group):
    # Plot TTFT and TPOT data seperately
    sub_plot(x_axis, vllm_ttft_data, apex_ttft_data, "TTFT", experiment_group)
    sub_plot(x_axis, vllm_tpot_data, apex_tpot_data, "TPOT", experiment_group)
    

def run_and_plot(experiment_group):
    # Get all the vllm experiments ran in the latest set of experiments
    specific_group = experiment_group
    filtered_df = df[df['experiment_group'] == specific_group]

    print("Analyzing experiment group: ", experiment_group)
    
    vllm_tpot_list = []
    vllm_ttft_list = []
    apex_tpot_list = []
    apex_ttft_list = []
    x_axis_vals = []


    # Run an APEX experiment for the latest vllm contingent traces
    for ind in filtered_df.index:
        ttft,tpot = run_apex(df['vllm_to_apex_trace'][ind],df['model'][ind],df['num_nodes'][ind],df['num_gpus'][ind])
        apex_ttft_list.append(ttft)
        apex_tpot_list.append(tpot)

        # Get vllm results from mapped files and convert to second
        vllm_ttft, vllm_tpot = get_vllm_results(df['vllm_metric_file'][ind], df['vllm_validation_file'][ind])
        vllm_ttft_list.append(vllm_ttft)
        vllm_tpot_list.append(vllm_tpot)
        x_axis_vals.append(df['num_gpus'][ind])

    
    apex_first_ttft = apex_ttft_list[0]
    apex_first_tpot = apex_tpot_list[0]
    vllm_first_ttft = vllm_ttft_list[0]
    vllm_first_tpot = vllm_tpot_list[0]

    
    # Normalize results
    normalized_apex_ttft_list = [apex_first_ttft / value for value in apex_ttft_list]
    normalized_apex_tpot_list = [apex_first_tpot / value for value in apex_tpot_list]
    normalized_vllm_ttft_list = [vllm_first_ttft / value  for value in vllm_ttft_list]
    normalized_vllm_tpot_list = [vllm_first_tpot / value for value in vllm_tpot_list]

    # Plot results
    plot(x_axis_vals, normalized_vllm_ttft_list, normalized_vllm_tpot_list, normalized_apex_ttft_list, normalized_apex_tpot_list, experiment_group)


def main(experiment_group):
    run_and_plot(experiment_group)


if __name__ == "__main__":

    # Replace 'your_file.csv' with the path to your CSV file
    df = pd.read_csv('./data_mappings.csv')
    # Get the last row's value of the 'experiment_group' column
    last_experiment_group = df['experiment_group'].iloc[-1]

    main(last_experiment_group)

