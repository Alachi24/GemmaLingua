# gemTun — Model Evaluation Plan (base vs fine-tuned)

> Local working doc. Goal: measure whether the fine-tuned adapters beat the base
> model on Igbo, and whether the chained training stages cost any earlier ability.
> No code written yet — this is the plan to execute next.

---

## 0. Confirmed facts (from the codebase, not assumptions)

- **Base model:** `google/gemma-4-E4B-it` — **Gemma 4 E4B, instruction-tuned**.
  Confirmed in `config/pipeline.yaml:21` and all training scripts
  (`train_lm.py:17`, `train_translation.py:20`, `train_masking.py:25`, `cache_model.py:14`).
  (NOTE: this is Gemma **4**, not Gemma 3n — earlier confusion corrected.)

- **Loading (`python/utils/model.py`):** 4-bit quantization (`load_in_4bit`,
  `bnb_4bit_compute_dtype=bfloat16`), `device_map="auto"`, `AutoModelForCausalLM`.
  LoRA: `r=16, alpha=32, dropout=0.05, target_modules="all-linear"`,
  excludes vision/audio towers.

- **Training uses RAW TEXT, not the chat template.** Despite the `-it` base, all
  scripts build a plain `"text"` field and feed `SFTTrainer` — no `apply_chat_template`.
  The models learned these exact prefixes:
  - **LM:** raw line of Igbo text (no prompt).
  - **Translation** (`train_translation.py:48`):
    ```
    Translate English to Igbo.
    English: {src}
    Igbo: {tgt}
    ```
  - **Masking** (`train_masking.py:58`):
    ```
    Fill in the blanks in the Igbo sentence.
    Igbo: {masked}
    Answer: {answer}
    ```

- **Adapters are CUMULATIVE / chained** (`load_adapter_for_training`, `model.py:54`;
  confirmed in run logs):

  | Adapter | Actually contains |
  |---|---|
  | `igbo-gemma4-lm-finetuned` | base + LM |
  | `igbo-gemma4-finetuned` | base + LM + translation |
  | `igbo-gemma4-masking-finetuned` | base + LM + translation + masking (**final model**) |

---

## Training results recap (from run logs)

| Stage | Adapter out | Runtime | train_loss (avg) | recent loss | token acc (final) | tokens | epochs |
|---|---|---|---|---|---|---|---|
| LM | `igbo-gemma4-lm-finetuned` | ~14.52 h | 2.105 | ~1.97 | **0.5989 (~59%)** | 65.73 M | 3 |
| Translation | `igbo-gemma4-finetuned` | ~2.43 h | 1.42 | ~1.05 | **0.7705 (~77%)** | 1.895 M | 3 |
| Masking | `igbo-gemma4-masking-finetuned` | ~2.28 h | 1.49 | ~1.13 | **0.7595 (~76%)** | 1.443 M | 3 |

- LM recent loss ~1.97 → perplexity ≈ e^1.97 ≈ **7.2**.
- `entropy ≈ loss` throughout LM stage → model is **well-calibrated** (not over/under-confident).
- Loss values are **NOT comparable across stages** (different objectives/token distributions).

### Open question this plan answers: "Is 59% good?"
- 59% = `mean_token_accuracy` = **top-1 next-token accuracy**, an *intrinsic* metric.
- Normal/healthy range for free-text causal LM is ~50–70%; 59% on low-resource Igbo is fine.
- **No published apples-to-apples number exists** for Gemma 4 E4B on Igbo next-token accuracy.
  Published Igbo numbers (IrokoBench, AfroBench) are *downstream task* accuracy — a
  different metric, NOT comparable to 59%.
- **The only valid baseline is to run the base model ourselves on our held-out data.**

---

## 1. Precondition: held-out data (without this, nothing is valid)

- Eval examples MUST be excluded from training.
- Translation + masking data both come from the translation CSV `ig` column, so a
  single held-out slice of those CSVs serves both.
- LM data is raw Igbo text/CSV — needs its own held-out slice.
- Hold out a fixed split (last ~5–10%) BEFORE anything; verify those exact sentences
  are not in the training files.
- Generate the masking eval from the **held-out** sentences using the same
  `load_masking_datasets` logic (same format, unseen content).

**TODO decisions:**
- [ ] Carve held-out split from existing CSVs/text under `data/`, or use a separate test set?
- [ ] Are the `en`/`ig` CSV references gold human translations (chrF/BLEU meaningful) or synthetic?

---

## 2. Comparisons to run

| # | Comparison | Purpose |
|---|---|---|
| A | base vs `lm-finetuned` | Did LM fine-tuning improve Igbo modeling? (answers "is 59% good?") |
| B | base (few-shot) vs `igbo-gemma4-finetuned` (0-shot) — translation | Real translator? |
| C | base (few-shot) vs `masking-finetuned` — infilling | Real infiller? |
| D | `finetuned` vs `masking-finetuned` on **translation** | Did masking stage damage translation? (forgetting) |
| E | `lm-finetuned` vs `masking-finetuned` on **LM perplexity** | Did later stages damage base Igbo modeling? |

A/B/C = "better than base?"  D/E = "did chaining cost us anything?" (the more interesting one).

---

## 3. Metrics per task

- **LM (intrinsic):** perplexity (= e^loss) + token accuracy on held-out Igbo.
  Identical to training metric → finally gives 59% a baseline.
- **Translation:** **chrF / chrF++** (lead — char-level, fair for morphologically rich Igbo)
  + **BLEU** (convention). Optional **COMET** (learned semantic). Generate from the
  `Translate English to Igbo...\nIgbo:` prompt; compare to gold `ig`.
- **Masking:** **span exact-match %** + **token accuracy on blanked spans only**
  (do NOT score copied context). Optional chrF of reconstructed answer.

---

## 4. Fairness rules (so the base isn't strawmanned)

1. Same **raw prompt format** for base and fine-tuned (training strings above) — NOT the chat template.
2. **Few-shot the base** (3–5 in-format examples); fine-tuned runs 0-shot. Note the asymmetry.
3. Same **decoding settings** for both (greedy or fixed temp/seed), same max-new-tokens.
4. Same **4-bit load** for both (match training).
5. Same **held-out examples** across all models in a given task.

---

## 5. Deliverable shape

```
Translation (held-out en→ig, N=500, greedy)
  base gemma-4-E4B-it (5-shot):   chrF 30.1   BLEU 7.9
  +trans adapter (0-shot):        chrF 48.3   BLEU 19.4   ← lift vs base
  +masking adapter (0-shot):      chrF 46.0   BLEU 18.1   ← forgetting check vs +trans

LM (held-out Igbo, N=...)
  base:              ppl 11.x   acc 0.5x
  +lm adapter:       ppl  7.2   acc 0.59
  +masking adapter:  ppl  ?     acc  ?       ← did later stages hurt LM
```

---

## 6. Optional / higher-effort

- **IrokoBench / AfroBench** Igbo tasks (AfriMMLU, AfriXNLI) on base vs final —
  real downstream ability, not just text modeling.
- **LLM-as-judge or human review** on ~50–100 translation/infill outputs —
  catches fluency/adequacy that chrF misses (important for low-resource langs).

---

## 7. Side issue noted (not eval, but open)

- `cache-base-model` stage exits 1 (77ms, first stage). Non-blocking — all training +
  export still succeeded — but it's why the pipeline logs
  "Pipeline finished with failures." Worth making it a clean no-op when the model is
  already cached. Separate task from eval.

---

## Next actions for tomorrow

1. Decide held-out data source (§1 TODOs).
2. Confirm whether translation references are gold.
3. Build eval scripts (mirror `model.py` loading: 4-bit, raw prompts) for comparisons A–E.
4. (Optional) wire eval as a pipeline stage.
