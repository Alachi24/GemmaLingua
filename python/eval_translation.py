"""
Evaluate translation adapter: chrF++ / BLEU on held-out en->ig pairs.
Compares base model (few-shot) vs fine-tuned adapters (0-shot).

Usage:
  source venv/bin/activate
  python python/eval_translation.py --test-data data/raw/igbo_translation_test.csv --train-data data/raw/igbo_translation_train.csv --adapters models/adapters/igbo-gemma4-finetuned
"""
import argparse
import random
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from utils.model import build_quantization_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate translation chrF++ / BLEU on held-out en->ig pairs"
    )
    parser.add_argument("--model-id", default="google/gemma-4-E4B-it")
    parser.add_argument("--test-data", required=True, help="Held-out translation CSV (en,ig columns)")
    parser.add_argument("--train-data", default=None,
                        help="Training CSV for few-shot examples (base model only)")
    parser.add_argument("--adapters", nargs="*", default=[],
                        help="Adapter paths to evaluate (none = base only)")
    parser.add_argument("--source-col", default="en")
    parser.add_argument("--target-col", default="ig")
    parser.add_argument("--source-name", default="English")
    parser.add_argument("--target-name", default="Igbo")
    parser.add_argument("--few-shot-n", type=int, default=3,
                        help="Number of few-shot examples for the base model")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--sample", type=int, default=None,
                        help="Limit evaluation to N test examples (useful for speed)")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_prompt(source, args, few_shot_examples=None):
    prompt = ""
    if few_shot_examples:
        for ex in few_shot_examples:
            prompt += (
                f"Translate {args.source_name} to {args.target_name}.\n"
                f"{args.source_name}: {ex[args.source_col]}\n"
                f"{args.target_name}: {ex[args.target_col]}\n\n"
            )
    prompt += (
        f"Translate {args.source_name} to {args.target_name}.\n"
        f"{args.source_name}: {source}\n"
        f"{args.target_name}:"
    )
    return prompt


def load_few_shot_examples(train_path, args):
    df = pd.read_csv(train_path).dropna()
    rng = random.Random(args.seed)
    return rng.sample(list(df.to_dict(orient="index").values()), args.few_shot_n)


@torch.no_grad()
def translate_all(model, tokenizer, sources, args, few_shot_examples=None):
    model.eval()
    outputs = []
    for src in sources:
        prompt = build_prompt(src, args, few_shot_examples)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        outputs.append(generated)
    return outputs


def compute_metrics(references, hypotheses):
    import evaluate
    chrf = evaluate.load("chrf")
    bleu = evaluate.load("sacrebleu")
    chrf_result = chrf.compute(predictions=hypotheses, references=[[r] for r in references],
                                word_order=2)
    bleu_result = bleu.compute(predictions=hypotheses, references=[[r] for r in references])
    return {
        "chrF++": chrf_result["score"],
        "BLEU": bleu_result["score"],
    }


def load_base_model(model_id, hf_token):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=build_quantization_config(),
        device_map="auto",
        token=hf_token,
    )
    model.config.use_cache = True
    return model


def main():
    args = parse_args()

    test_df = pd.read_csv(args.test_data).dropna()
    if args.sample:
        test_df = test_df.sample(n=args.sample, random_state=args.seed)
    sources = test_df[args.source_col].astype(str).tolist()
    references = test_df[args.target_col].astype(str).tolist()
    print(f"Loaded {len(sources)} test examples\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    few_shot_examples = None
    if args.train_data:
        few_shot_examples = load_few_shot_examples(args.train_data, args)
        print(f"Loaded {len(few_shot_examples)} few-shot examples from training data")

    results = {}

    if args.train_data:
        print("Evaluating base model (few-shot)...")
        model = load_base_model(args.model_id, args.hf_token)
        hypotheses = translate_all(model, tokenizer, sources, args, few_shot_examples)
        metrics = compute_metrics(references, hypotheses)
        results["base (few-shot)"] = metrics
        print(f"  chrF++: {metrics['chrF++']:.2f}   BLEU: {metrics['BLEU']:.2f}")
        del model
    else:
        print("Skipping base model (no --train-data)")

    for adapter_path in args.adapters:
        label = adapter_path.rstrip("/").split("/")[-1]
        print(f"Evaluating adapter [{label}] (0-shot)...")
        model = load_base_model(args.model_id, args.hf_token)
        model = PeftModel.from_pretrained(model, adapter_path)
        hypotheses = translate_all(model, tokenizer, sources, args, few_shot_examples=None)
        metrics = compute_metrics(references, hypotheses)
        results[label] = metrics
        print(f"  chrF++: {metrics['chrF++']:.2f}   BLEU: {metrics['BLEU']:.2f}")
        del model

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"{'Model':<32} {'chrF++':<14} {'BLEU':<14}")
    print(sep)
    for label, m in results.items():
        print(f"{label:<32} {m['chrF++']:<14.2f} {m['BLEU']:<14.2f}")
    print(sep)


if __name__ == "__main__":
    main()
