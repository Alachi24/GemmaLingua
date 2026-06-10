"""
Downloads a translation dataset from Hugging Face and saves each split as a CSV
with the project's source/target column names (ready for the pipeline).

Language-agnostic: point --dataset at any HF translation dataset and map its
columns to the pipeline's source/target columns.

Usage:
  python download_translation.py \
      --dataset iamwille/igbo-translation \
      --src-hf-col English --tgt-hf-col Igbo \
      --source-col en --target-col ig \
      --prefix igbo_translation \
      [--output ./data/raw] [--hf-token <token>]
"""
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Download a translation dataset from Hugging Face")
    parser.add_argument("--dataset", default="iamwille/igbo-translation",
                        help="Hugging Face dataset id")
    parser.add_argument("--src-hf-col", default="English",
                        help="Source column name in the HF dataset")
    parser.add_argument("--tgt-hf-col", default="Igbo",
                        help="Target column name in the HF dataset")
    parser.add_argument("--source-col", default="en",
                        help="Source column name to write in the output CSV")
    parser.add_argument("--target-col", default="ig",
                        help="Target column name to write in the output CSV")
    parser.add_argument("--prefix", default="igbo_translation",
                        help="Output filename prefix: <prefix>_<split>.csv")
    parser.add_argument("--output", default="./data/raw",
                        help="Output directory for CSV files")
    parser.add_argument("--hf-token", default=None,
                        help="Hugging Face token (for gated datasets)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    token = args.hf_token or os.environ.get("HF_TOKEN")

    print(f"Loading {args.dataset} from Hugging Face...")
    from datasets import load_dataset

    dataset = load_dataset(args.dataset, token=token)

    first_path = None
    for split in dataset:
        df = dataset[split].to_pandas()
        df = df.rename(columns={args.src_hf_col: args.source_col, args.tgt_hf_col: args.target_col})
        df = df[[args.source_col, args.target_col]]
        path = os.path.join(output_dir, f"{args.prefix}_{split}.csv")
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} rows to {path}")
        if first_path is None:
            first_path = path

    print(f"\nDone! Add this to your pipeline.yaml:\n")
    print(f"data:")
    print(f"  translation_files:")
    print(f"    - {first_path}")


if __name__ == "__main__":
    main()
