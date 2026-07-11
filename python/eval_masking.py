"""
Evaluate masking (span-infilling) adapter: span exact-match + token accuracy.
Compares base model (few-shot) vs fine-tuned adapters (0-shot).

Usage:
  source venv/bin/activate
  python python/eval_masking.py --data data/raw/igbo_translation_test.csv --train-data data/raw/igbo_translation_train.csv --adapters models/adapters/igbo-gemma4-masking-finetuned
"""
import argparse
import random
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from utils.data import _mask_sentence, MASK_TOKEN, ANSWER_SEP
from utils.model import build_quantization_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate span-infilling: exact-match + token accuracy"
    )
    parser.add_argument("--model-id", default="google/gemma-4-E4B-it")
    parser.add_argument("--data", required=True, nargs="+",
                        help="CSV(s) with the target column to mask")
    parser.add_argument("--train-data", default=None,
                        help="Training CSV for few-shot examples (base model only)")
    parser.add_argument("--adapters", nargs="*", default=[])
    parser.add_argument("--target-col", default="ig")
    parser.add_argument("--target-name", default="Igbo")
    parser.add_argument("--few-shot-n", type=int, default=3)
    parser.add_argument("--mask-rate", type=float, default=0.15)
    parser.add_argument("--mean-span", type=int, default=3)
    parser.add_argument("--max-spans", type=int, default=5)
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_prompt(masked_sentence, args, few_shot_examples=None):
    prompt = ""
    if few_shot_examples:
        for masked, answer in few_shot_examples:
            prompt += (
                f"Fill in the blanks in the {args.target_name} sentence.\n"
                f"{args.target_name}: {masked}\n"
                f"Answer: {answer}\n\n"
            )
    prompt += (
        f"Fill in the blanks in the {args.target_name} sentence.\n"
        f"{args.target_name}: {masked_sentence}\n"
        f"Answer:"
    )
    return prompt


def load_masked_examples(filepaths, args, train=False):
    rng = random.Random(args.seed)
    rows = []
    for fp in filepaths:
        df = pd.read_csv(fp).dropna(subset=[args.target_col])
        for text in df[args.target_col].astype(str):
            result = _mask_sentence(text, rng, args.mask_rate, args.mean_span,
                                    args.max_spans, args.min_words)
            if result is None:
                continue
            masked, answer = result
            rows.append({"masked": masked, "answer": answer, "original": text})
    return rows


def load_few_shot_examples(train_path, args):
    examples = load_masked_examples([train_path], args)
    rng = random.Random(args.seed)
    selected = rng.sample(examples, min(args.few_shot_n, len(examples)))
    return [(ex["masked"], ex["answer"]) for ex in selected]


@torch.no_grad()
def generate_answers(model, tokenizer, prompts, max_new_tokens):
    model.eval()
    outputs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        outputs.append(generated)
    return outputs


def parse_answer(text):
    idx = text.find("Answer:")
    ans = text[idx + len("Answer:"):].strip() if idx != -1 else text.strip()
    return ans


def compute_metrics(gold_answers, pred_answers):
    exact_matches = 0
    total_tokens_gold = 0
    total_tokens_correct = 0
    n = len(gold_answers)

    for gold, pred in zip(gold_answers, pred_answers):
        gold_parsed = parse_answer(gold) if ANSWER_SEP in gold else gold
        pred_parsed = parse_answer(pred) if ANSWER_SEP in pred else pred

        if gold_parsed == pred_parsed:
            exact_matches += 1

        gold_tokens = gold_parsed.replace(ANSWER_SEP, " ").split()
        pred_tokens = pred_parsed.replace(ANSWER_SEP, " ").split()
        max_len = max(len(gold_tokens), len(pred_tokens))
        for i in range(min(len(gold_tokens), len(pred_tokens))):
            if gold_tokens[i] == pred_tokens[i]:
                total_tokens_correct += 1
        total_tokens_gold += len(gold_tokens)

    exact_match_rate = exact_matches / n if n else 0.0
    token_acc = total_tokens_correct / total_tokens_gold if total_tokens_gold else 0.0
    return {
        "exact_match": exact_match_rate,
        "token_accuracy": token_acc,
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

    test_examples = load_masked_examples(args.data, args)
    if args.sample:
        rng = random.Random(args.seed)
        test_examples = rng.sample(test_examples, min(args.sample, len(test_examples)))
    print(f"Loaded {len(test_examples)} masking test examples\n")

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
        prompts = [build_prompt(ex["masked"], args, few_shot_examples) for ex in test_examples]
        gold_answers = [ex["answer"] for ex in test_examples]
        model = load_base_model(args.model_id, args.hf_token)
        pred_answers = generate_answers(model, tokenizer, prompts, args.max_new_tokens)
        metrics = compute_metrics(gold_answers, pred_answers)
        results["base (few-shot)"] = metrics
        print(f"  exact-match: {metrics['exact_match']:.4f}   token-acc: {metrics['token_accuracy']:.4f}")
        del model
    else:
        print("Skipping base model (no --train-data)")

    for adapter_path in args.adapters:
        label = adapter_path.rstrip("/").split("/")[-1]
        print(f"Evaluating adapter [{label}] (0-shot)...")
        prompts = [build_prompt(ex["masked"], args) for ex in test_examples]
        gold_answers = [ex["answer"] for ex in test_examples]
        model = load_base_model(args.model_id, args.hf_token)
        model = PeftModel.from_pretrained(model, adapter_path)
        pred_answers = generate_answers(model, tokenizer, prompts, args.max_new_tokens)
        metrics = compute_metrics(gold_answers, pred_answers)
        results[label] = metrics
        print(f"  exact-match: {metrics['exact_match']:.4f}   token-acc: {metrics['token_accuracy']:.4f}")
        del model

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"{'Model':<32} {'Exact Match':<14} {'Token Acc':<14}")
    print(sep)
    for label, m in results.items():
        print(f"{label:<32} {m['exact_match']:<14.4f} {m['token_accuracy']:<14.4f}")
    print(sep)


if __name__ == "__main__":
    main()
