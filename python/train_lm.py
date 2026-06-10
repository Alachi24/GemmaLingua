"""
Language-model fine-tuning for Gemma 4 E4B on raw Igbo text.
Usage: python train_lm.py --data <path> [--data <path> ...] --output <dir> [options]
"""
import argparse
import os
import sys
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from utils.data import load_lm_datasets
from utils.model import load_tokenizer_and_model, build_lora_config, apply_lora


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 E4B on Igbo text")
    parser.add_argument("--model-id", default="google/gemma-4-E4B-it")
    parser.add_argument("--data", required=True, nargs="+", help="Path(s) to Igbo .txt or .csv file")
    parser.add_argument("--output", default="./models/adapters/igbo-gemma4-lm-finetuned")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-strategy", default="epoch")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()

    lines = load_lm_datasets(args.data)
    dataset = Dataset.from_list([{"text": line} for line in lines])
    print("Dataset ready!")
    print("Sample:", dataset[0])

    tokenizer, model = load_tokenizer_and_model(args.model_id, args.hf_token)

    lora_config = build_lora_config(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model = apply_lora(model, lora_config)

    sft_config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        bf16=args.bf16,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=dataset,
    )

    print("Starting language model training...")
    trainer.train()
    print("Training complete!")

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
