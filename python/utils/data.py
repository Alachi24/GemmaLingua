import os
import pandas as pd


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
