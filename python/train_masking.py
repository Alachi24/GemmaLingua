"""
Span-infilling (masking) fine-tuning for Gemma 4 E4B on Igbo sentences.

Multi-span blanks are derived from the target column of translation CSV(s):
random word spans are replaced with a blank marker and the model learns to
generate the missing spans (causal infilling -- no MLM head needed).

Usage: python train_masking.py --data <csv> [--data <csv> ...] --output <dir> [options]
"""
import argparse
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from utils.data import load_masking_datasets
from utils.model import (
    load_tokenizer_and_model,
    build_lora_config,
    apply_lora,
    load_adapter_for_training,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 E4B on span-infilling")
    parser.add_argument("--model-id", default="google/gemma-4-E4B-it")
    parser.add_argument("--data", required=True, nargs="+", help="Path(s) to CSV with the target column")
    parser.add_argument("--output", default="./models/adapters/igbo-gemma4-masking-finetuned")
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
    # Language config -- only target-col/target-name are used here, but source-*
    # are accepted for compatibility with the shared Java arg builder.
    parser.add_argument("--source-col", default="en")
    parser.add_argument("--source-name", default="English")
    parser.add_argument("--target-col", default="ig", help="CSV column to mask")
    parser.add_argument("--target-name", default="Igbo", help="Target language name used in the prompt")
    # Masking knobs.
    parser.add_argument("--mask-rate", type=float, default=0.15, help="Fraction of words blanked per sentence")
    parser.add_argument("--mean-span", type=int, default=3, help="Average blanked-span length (words)")
    parser.add_argument("--max-spans", type=int, default=5, help="Max number of blanks per sentence")
    # Chaining: continue training an existing adapter (e.g. the translation adapter).
    parser.add_argument("--init-adapter", default=None,
                        help="Path to an existing adapter to load and keep training")
    return parser.parse_args()


def make_formatter(args):
    def format_row(row):
        return {
            "text": (
                f"Fill in the blanks in the {args.target_name} sentence.\n"
                f"{args.target_name}: {row['masked']}\n"
                f"Answer: {row['answer']}"
            )
        }
    return format_row


def main():
    args = parse_args()

    rows = load_masking_datasets(
        args.data,
        target_col=args.target_col,
        mask_rate=args.mask_rate,
        mean_span=args.mean_span,
        max_spans=args.max_spans,
    )
    format_row = make_formatter(args)
    dataset = Dataset.from_list([format_row(row) for row in rows])
    print("Dataset ready!")
    print(dataset[0])

    tokenizer, model = load_tokenizer_and_model(args.model_id, args.hf_token)

    if args.init_adapter:
        # Continue training the translation adapter on masked data.
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

    print("Starting masking training...")
    trainer.train()
    print("Training complete!")

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()