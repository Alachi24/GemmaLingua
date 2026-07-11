"""
Evaluate LM adapter: perplexity + token accuracy on held-out Igbo text.
Compares base model vs one or more fine-tuned adapters.

Usage:
  source venv/bin/activate
  python python/eval_lm.py --data data/raw/monolingual/*.txt --adapters models/adapters/igbo-gemma4-lm-finetuned
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from utils.data import load_lm_datasets
from utils.model import build_quantization_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LM perplexity + token accuracy on held-out Igbo text"
    )
    parser.add_argument("--model-id", default="google/gemma-4-E4B-it")
    parser.add_argument("--data", required=True, nargs="+",
                        help="Held-out Igbo .txt or .csv file(s)")
    parser.add_argument("--adapters", nargs="*", default=[],
                        help="Adapter paths to evaluate (none = base only)")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max token length per example")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, tokenizer, texts, max_length):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        labels = inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(**inputs, labels=labels)
        logits = outputs.logits[..., :-1, :].contiguous()
        labels_shifted = labels[..., 1:].contiguous()

        mask = labels_shifted != -100
        n = mask.sum().item()
        if n == 0:
            continue

        total_loss += outputs.loss.item() * n
        total_tokens += n

        preds = logits.argmax(dim=-1)
        total_correct += ((preds == labels_shifted) & mask).sum().item()

    avg_loss = total_loss / total_tokens if total_tokens else 0.0
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    acc = total_correct / total_tokens if total_tokens else 0.0
    return ppl, acc


def load_base_model(model_id, hf_token):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=build_quantization_config(),
        device_map="auto",
        token=hf_token,
    )
    model.config.use_cache = True
    return model


def evaluate_all(all_texts, max_length, model_id, adapters, hf_token):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    results = {}

    print("Evaluating base model (no adapter)...")
    model = load_base_model(model_id, hf_token)
    ppl, acc = evaluate(model, tokenizer, all_texts, max_length)
    results["base"] = {"perplexity": ppl, "accuracy": acc}
    print(f"  perplexity: {ppl:.4f}   token accuracy: {acc:.4f}")
    del model

    for adapter_path in adapters:
        label = adapter_path.rstrip("/").split("/")[-1]
        print(f"Evaluating adapter [{label}]...")
        model = load_base_model(model_id, hf_token)
        model = PeftModel.from_pretrained(model, adapter_path)
        ppl, acc = evaluate(model, tokenizer, all_texts, max_length)
        results[label] = {"perplexity": ppl, "accuracy": acc}
        print(f"  perplexity: {ppl:.4f}   token accuracy: {acc:.4f}")
        del model

    return results


def print_table(results):
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"{'Model':<32} {'Perplexity':<14} {'Token Acc':<14}")
    print(sep)
    for label, m in results.items():
        print(f"{label:<32} {m['perplexity']:<14.4f} {m['accuracy']:<14.4f}")
    print(sep)


def main():
    args = parse_args()
    texts = load_lm_datasets(args.data)
    print(f"Loaded {len(texts)} held-out examples\n")
    results = evaluate_all(texts, args.max_length, args.model_id, args.adapters, args.hf_token)
    print_table(results)


if __name__ == "__main__":
    main()
