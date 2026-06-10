"""
Translation fine-tuning for Gemma 4 E4B on English-Igbo pairs.
Usage: python train_translation.py --data <path> [--data <path> ...] --output <dir> [options]
"""
import argparse
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from utils.data import load_translation_datasets
from utils.model import (
    load_tokenizer_and_model,
    build_lora_config,
    apply_lora,
    load_adapter_for_training,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 E4B on translation pairs")
    parser.add_argument("--model-id", default="google/gemma-4-E4B-it")
    parser.add_argument("--data", required=True, nargs="+", help="Path(s) to CSV with source/target columns")
    parser.add_argument("--output", default="./models/adapters/igbo-gemma4-finetuned")
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
    # Language config (drives prompt template + CSV columns) -- defaults to en->ig.
    parser.add_argument("--source-col", default="en", help="CSV column for source text")
    parser.add_argument("--target-col", default="ig", help="CSV column for target text")
    parser.add_argument("--source-name", default="English", help="Source language name used in the prompt")
    parser.add_argument("--target-name", default="Igbo", help="Target language name used in the prompt")
    # Chaining: continue training an existing adapter (e.g. the monolingual LM adapter).
    parser.add_argument("--init-adapter", default=None,
                        help="Path to an existing adapter to load and keep training")
    return parser.parse_args()


def make_formatter(args):
    def format_row(row):
        return {
            "text": (
                f"Translate {args.source_name} to {args.target_name}.\n"
                f"{args.source_name}: {row[args.source_col]}\n"
                f"{args.target_name}: {row[args.target_col]}"
            )
        }
    return format_row


def main():
    args = parse_args()

    df = load_translation_datasets(args.data, args.source_col, args.target_col)
    format_row = make_formatter(args)
    dataset = Dataset.from_list([format_row(row) for _, row in df.iterrows()])
    print("Dataset ready!")
    print(dataset[0])

    tokenizer, model = load_tokenizer_and_model(args.model_id, args.hf_token)

    if args.init_adapter:
        # Continue training the monolingual adapter on translation data.
        model = load_adapter_for_training(model, args.init_adapter)
    else:
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

    print("Starting translation training...")
    trainer.train()
    print("Training complete!")

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
