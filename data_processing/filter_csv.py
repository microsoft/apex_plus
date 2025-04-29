import pandas as pd
import argparse

def main(args):
    df = pd.read_csv(args.input_csv)
    df_filtered = df[df["ContextTokens"] <= args.max_tokens]
    df_filtered.to_csv(args.output_csv, index=False)
    
    print(f"Filtered CSV file '{args.output_csv}' has been saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter CSV based on ContextTokens column.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the filtered CSV file.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum allowed tokens for ContextTokens filter.")
    
    args = parser.parse_args()
    main(args)
