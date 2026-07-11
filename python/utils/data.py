import os
import random
import pandas as pd

MASK_TOKEN = "___"
ANSWER_SEP = " | "


def read_text_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".txt", ".csv"):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    raise ValueError(f"Unsupported file type: {ext}. Only .txt and .csv are supported.")


def load_lm_dataset(filepath):
    raw = read_text_file(filepath)
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    print(f"Loaded {len(lines)} Igbo examples from {filepath}")
    return lines


def load_lm_datasets(filepaths):
    all_lines = []
    for fp in filepaths:
        all_lines.extend(load_lm_dataset(fp))
    print(f"Total: {len(all_lines)} Igbo examples across {len(filepaths)} files")
    return all_lines


def load_translation_dataset(filepath, source_col="en", target_col="ig"):
    df = pd.read_csv(filepath)
    df = df.dropna()
    required = {source_col, target_col}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}. Found: {list(df.columns)}")
    print(f"Loaded {len(df)} translation pairs from {filepath}")
    return df


def load_translation_datasets(filepaths, source_col="en", target_col="ig"):
    dfs = []
    for fp in filepaths:
        dfs.append(load_translation_dataset(fp, source_col, target_col))
    merged = pd.concat(dfs, ignore_index=True).drop_duplicates()
    print(f"Total: {len(merged)} translation pairs across {len(filepaths)} files")
    return merged


def _mask_sentence(text, rng, mask_rate, mean_span, max_spans, min_words):
    """Blank out one or more non-overlapping word spans (multi-span infilling).

    Returns (masked_text, answer) where masked_text holds the sentence with each
    blanked span replaced by MASK_TOKEN, and answer is the removed spans joined
    by ANSWER_SEP in left-to-right order. Returns None if the sentence is too
    short to mask.
    """
    words = text.split()
    n = len(words)
    if n < min_words:
        return None

    budget = max(1, round(n * mask_rate))
    masked_idx = set()
    spans = []  # (start, length)
    attempts = 0
    while len(masked_idx) < budget and len(spans) < max_spans and attempts < 50:
        attempts += 1
        span_len = min(rng.randint(1, max(1, 2 * mean_span - 1)), n)
        start = rng.randint(0, n - span_len)
        idxs = set(range(start, start + span_len))
        # Skip overlapping or adjacent picks so blanks stay separate.
        if idxs & masked_idx:
            continue
        if (start - 1) in masked_idx or (start + span_len) in masked_idx:
            continue
        spans.append((start, span_len))
        masked_idx |= idxs

    if not spans:
        return None

    spans.sort()
    answers = [" ".join(words[s:s + l]) for s, l in spans]
    span_start = {s: l for s, l in spans}
    out, i = [], 0
    while i < n:
        if i in span_start:
            out.append(MASK_TOKEN)
            i += span_start[i]
        else:
            out.append(words[i])
            i += 1
    return " ".join(out), ANSWER_SEP.join(answers)


def load_masking_datasets(filepaths, target_col="ig", mask_rate=0.15,
                          mean_span=3, max_spans=5, min_words=4, seed=42):
    """Build multi-span infilling examples from the target column of CSV(s).

    Each row becomes a dict {"masked": ..., "answer": ...}. Sentences too short
    to mask are skipped. Masking is seeded for reproducibility.
    """
    rng = random.Random(seed)
    rows = []
    skipped = 0
    for fp in filepaths:
        df = pd.read_csv(fp)
        if target_col not in df.columns:
            raise ValueError(f"CSV must contain column '{target_col}'. Found: {list(df.columns)}")
        df = df.dropna(subset=[target_col]).drop_duplicates(subset=[target_col])
        for text in df[target_col].astype(str):
            result = _mask_sentence(text, rng, mask_rate, mean_span, max_spans, min_words)
            if result is None:
                skipped += 1
                continue
            masked, answer = result
            rows.append({"masked": masked, "answer": answer})
    print(f"Built {len(rows)} masking examples ({skipped} skipped) across {len(filepaths)} files")
    return rows
