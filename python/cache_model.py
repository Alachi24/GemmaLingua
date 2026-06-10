"""
Downloads Gemma 4 E4B from Hugging Face to a local cache directory.
Only downloads if the model is not already cached.

Usage: python cache_model.py --model-id <id> --cache-dir <dir> [--hf-token <token>]
"""
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Cache base model locally")
    parser.add_argument("--model-id", default="google/gemma-4-E4B-it")
    parser.add_argument("--cache-dir", default="./models/base")
    parser.add_argument("--hf-token", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cache_dir = os.path.abspath(args.cache_dir)
    model_path = os.path.join(cache_dir, args.model_id.replace("/", "--"))
    token = args.hf_token or os.environ.get("HF_TOKEN")

    if os.path.exists(os.path.join(model_path, "config.json")):
        print(f"Model already cached at {model_path}")
        return

    os.makedirs(model_path, exist_ok=True)

    print(f"Downloading {args.model_id} to {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=token,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )

    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)
    print(f"Model cached successfully at {model_path}")


if __name__ == "__main__":
    main()
