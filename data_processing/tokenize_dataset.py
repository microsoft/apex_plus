import pandas as pd
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer


def tokenize_and_count(example, tokenizer, context_column, target_column):
    article_tokens = tokenizer(
        example[context_column], truncation=True, padding=True, return_tensors="pt"
    )
    abstract_tokens = tokenizer(
        example[target_column], truncation=True, padding=True, return_tensors="pt"
    )

    example["article_token_count"] = len(article_tokens["input_ids"][0])
    example["abstract_token_count"] = len(abstract_tokens["input_ids"][0])

    return example

def main(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    tokenized_dataset = dataset.map(
        lambda example: tokenize_and_count(example, tokenizer, args.context_column, args.target_column)
    )

    df = pd.DataFrame(tokenized_dataset)
    df_selected = df[["article_token_count", "abstract_token_count"]]
    df_selected["TIMESTAMP"] = args.timestamp
    df_selected = df_selected.rename(
        columns={
            "article_token_count": "ContextTokens",
            "abstract_token_count": "GeneratedTokens",
        }
    )

    columns_order = ["TIMESTAMP", "ContextTokens", "GeneratedTokens"]
    df_selected = df_selected[columns_order]
    df_selected.to_csv(args.output_csv, index=False)
    print(f"CSV file has been saved to '{args.output_csv}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize dataset and save token counts.")
    parser.add_argument("--model", type=str, required=True, help="Model name or path on HuggingFace Hub.")
    parser.add_argument("--dataset_name", type=str, required=True, help="HuggingFace dataset name (e.g., 'ccdv/arxiv-summarization').")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset config name (if any, e.g., 'section').")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load (e.g., 'train', 'test', 'validation').")
    parser.add_argument("--context_column", type=str, default="article", help="Column name for context (input) text.")
    parser.add_argument("--target_column", type=str, default="abstract", help="Column name for target (output) text.")
    parser.add_argument("--timestamp", type=str, default="2023-11-16 18:15:46.6805900", help="Timestamp to assign to each row.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the output CSV file.")

    args = parser.parse_args()
    main(args)
