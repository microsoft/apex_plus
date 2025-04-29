import pandas as pd
import numpy as np
import math
import random
import argparse
from datetime import datetime, timedelta

# Gnerate the next timestamp based on Poisson distribution
def generate_poisson_timestamps(start_time, qps, num_timestamps):
    timestamps = []
    current_time = start_time

    for _ in range(num_timestamps):
        next_interval = -math.log(1.0 - random.random()) / qps
        current_time += timedelta(seconds=next_interval)
        timestamps.append(current_time)

    return timestamps

def main(args):
    df = pd.read_csv(args.input_csv)
    start_time = datetime.now()
    df["TIMESTAMP"] = generate_poisson_timestamps(start_time, args.qps, len(df))
    selected_columns = ["TIMESTAMP"] + args.selected_columns
    df_selected = df[selected_columns]
    df_selected.to_csv(args.output_csv, index=False)

    print(f"CSV file with generated timestamps saved to '{args.output_csv}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate timestamps and filter CSV columns.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the output CSV file.")
    parser.add_argument("--qps", type=float, default=0.5, help="Queries per second (rate) for Poisson timestamp generation.")
    parser.add_argument("--selected_columns", type=str, nargs="+", default=["ContextTokens", "GeneratedTokens"],
                        help="Additional columns to include along with TIMESTAMP.")

    args = parser.parse_args()
    main(args)

