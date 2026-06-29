# Prompt Injection Classifier — Performance Improvement Plan

> **Status**: Planned  
> **Baseline**: `all-MiniLM-L6-v2` (384-dim) + 16-dim lexical features → Residual MLP `[384→256→128]`

---

## Current Architecture Overview

```
Input Text
    │
    ├── SentenceTransformer Encoder (all-MiniLM-L6-v2)
    │       → 384-dim normalized embedding
    │
    ├── Handcrafted Lexical Features (16-dim)
    │       → keyword density, entropy, regex patterns, etc.
    │
    └── Concatenated (400-dim)
            │
         [MLP Head]
         Linear(400 → 384) + LayerNorm + GELU + Dropout(0.2)  [+ residual]
         Linear(384 → 256) + LayerNorm + GELU + Dropout(0.2)  [+ residual]
         Linear(256 → 128) + LayerNorm + GELU + Dropout(0.2)
         Linear(128 → 2)   logits
            │
         Softmax → injection_prob
            │
         Calibrated threshold → benign / injection / uncertain
```

**Training pipeline**:
1. Phase 1 — balanced training with OneCycleLR + label smoothing + class weights  
2. Phase 2 — hard negative fine-tuning (false-positive benign samples upsampled ×3)  
3. Threshold calibration — F1-sweep over val set to find optimal decision boundary  

---

## Improvements (Ranked by Impact)

---

### 🔴 HIGH IMPACT

---

#### 1. Upgrade Encoder: `all-MiniLM-L6-v2` → `all-mpnet-base-v2`

| Property | MiniLM-L6-v2 | mpnet-base-v2 |
|---|---|---|
| Parameters | 22M | 110M |
| Embedding dim | 384 | 768 |
| SBERT BEIR score | ~0.60 | ~0.70 |
| Encoding speed | ~5× faster | Baseline |

Since embeddings are **disk-cached** in `data/embed_cache/`, the slower encoding is a **one-time cost** per dataset regeneration.

**Changes required**:
- `config.py`: `EMBEDDING_MODEL = "all-mpnet-base-v2"`, `EMBEDDING_DIM = 768`
- `config.py`: `MLP_INPUT_DIM = 768 + 24 = 792`
- `config.py`: `HIDDEN_DIMS = [512, 384, 256, 128]` (deeper to handle richer input)
- Delete existing embed cache so it re-encodes with the new model

---

#### 2. Expand Lexical Features: 16-dim → 24-dim

Add **8 new handcrafted features** targeting attack patterns not covered by the current 16:

| # | Feature Name | What it Captures |
|---|---|---|
| f17 | `instruction_hierarchy_spoof` | `SYSTEM:`, `USER:`, `ASSISTANT:` role-spoofing delimiters |
| f18 | `nested_brackets` | `[[...]]`, `{{...}}` deep nesting (indirect injection staging) |
| f19 | `token_smuggling` | Zero-width spaces, invisible Unicode chars, homoglyphs |
| f20 | `multilingual_switch` | Abrupt script change mid-text (UTF script detection) |
| f21 | `sentence_length_variance` | High variance = long instructions hidden after short decoys |
| f22 | `number_density` | Dense numeric content correlates with config/parameter overrides |
| f23 | `prompt_framing_phrase` | "Answer the following", "Given the above", "Based on this" |
| f24 | `ellipsis_separator` | `...` / `---` / `===` separators used to hide injected content |

**Changes required**:
- `feature_engineering.py`: Implement `f17`–`f24`, update return list to 24 elements
- `config.py`: `LEXICAL_FEATURE_DIM = 24`

---

#### 3. Training-Time Text Augmentation

Augment **injection samples only** at dataset-generation time to expose the model to obfuscation variants:

| Technique | Example |
|---|---|
| Case flipping | `"Ignore"` → `"iGnOrE"` / `"IGNORE"` |
| Whitespace injection | `"ignore"` → `"i g n o r e"` |
| Leet-speak | `"ignore"` → `"1gn0r3"` |
| Synonym swap | `"ignore"` → `"disregard"` / `"bypass"` |
| Character homoglyph | `"o"` → `"0"`, `"l"` → `"1"` |

Applied at generation time in `generate_data.py`, keeping the cache invalidation system intact.

---

### 🟡 MEDIUM IMPACT

---

#### 4. Mixup Training

[Mixup (Zhang et al., 2018)](https://arxiv.org/abs/1710.09412) interpolates between pairs of training samples and their labels:

```
x̃ = λ·xᵢ + (1−λ)·xⱼ
ỹ = λ·yᵢ + (1−λ)·yⱼ,  λ ~ Beta(α, α), α=0.2
```

Applied in `train_epoch()` over pre-computed embeddings. Acts as a strong regularizer:
- Prevents overconfident predictions at class boundaries
- Improves calibration of `injection_prob` scores
- No extra data needed — applied on-the-fly per batch

**Changes required**:
- `train.py`: Add `mixup_batch()` helper, call inside `train_epoch()` when `MIXUP_ALPHA > 0`
- `config.py`: `MIXUP_ALPHA = 0.2`

---

#### 5. Extended Hard Negative Mining (Phase 2)

Current Phase 2 only mines **false positives** (benign predicted as injection).  
Extend to also mine **false negatives** (injections predicted as benign — the more dangerous error):

```
Phase 2 dataset = train_data
                + benign FP samples × 3   (current)
                + injection FN samples × 5  (NEW — missed attacks are worse)
```

The higher upsample factor (×5) reflects the asymmetric cost: a missed attack is more harmful than a false alarm.

**Changes required**:
- `train.py`: Update `mine_hard_negatives()` to return both FP benign and FN injection samples
- `config.py`: Add `HARD_FN_UPSAMPLE = 5`

---

#### 6. Focal Loss

[Focal Loss (Lin et al., 2017)](https://arxiv.org/abs/1708.02002) down-weights easy examples and concentrates gradient on hard ones:

```
FL(p) = −α · (1 − pₜ)^γ · log(pₜ)
```

With `γ=2, α=0.25` this automatically focuses on the hard negatives the model struggles with, without explicit mining. Can be used **instead of** or **together with** hard negative mining.

**Changes required**:
- `train.py`: Implement `FocalLoss` module, use as `criterion` in Phase 1
- `config.py`: `FOCAL_GAMMA = 2.0`, `FOCAL_ALPHA = 0.25`

---

### 🟢 LOWER IMPACT (Polish & Calibration)

---

#### 7. Temperature Scaling (Post-hoc Calibration)

After all training phases, learn a single scalar temperature `T` that calibrates the softmax:

```
p_calibrated = softmax(logits / T)
```

Minimise NLL on the validation set to find `T`. This replaces the manual threshold sweep with a principled calibration and improves the reliability of the `injection_prob` confidence scores.

**Changes required**:
- `model.py`: Add `TemperatureScaler` wrapper module
- `train.py`: Add temperature calibration step after Phase 2, save `T` to checkpoint
- `predict.py`: Apply temperature scaling at inference

---

#### 8. 3-Seed Ensemble

Train 3 models with different random seeds (42, 7, 123). At inference, average their `injection_prob`:

```
p_ensemble = (p₁ + p₂ + p₃) / 3
```

For a 400-dim MLP, inference is microseconds per sample — 3× latency is negligible.  
Empirically reduces variance by **~1–2 F1 points** with no other changes.

---

## Summary Table

| # | Improvement | Files Changed | Expected Gain | Effort |
|---|---|---|---|---|
| 1 | Encoder upgrade (MiniLM → MPNet) | `config.py`, re-encode cache | High | Low (one-time) |
| 2 | Lexical features 16→24 | `feature_engineering.py`, `config.py` | Medium-High | Medium |
| 3 | Text augmentation at data-gen | `generate_data.py` | Medium | Medium |
| 4 | Mixup training | `train.py`, `config.py` | Medium | Low |
| 5 | Extended hard neg mining (FN too) | `train.py`, `config.py` | Medium | Low |
| 6 | Focal Loss | `train.py`, `config.py` | Medium | Low |
| 7 | Temperature scaling | `model.py`, `train.py`, `predict.py` | Low-Medium | Low |
| 8 | 3-seed ensemble | `train.py`, `predict.py` | Low-Medium | Medium |

---

## Verification

After each change, run:

```bash
python -m prompt_injection.train
```

Compare the final `classification_report` against the baseline checkpoint. Key metrics:

| Metric | What it means |
|---|---|
| `injection recall` | Fraction of real attacks caught — **most critical** |
| `benign precision` | False alarm rate — should stay high |
| `macro-F1` | Overall balance between both classes |
| `calibrated threshold` | Should be close to 0.5 for a well-calibrated model |

---

## References

- [Mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412) — Zhang et al., 2018
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) — Lin et al., 2017
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) — Guo et al., 2017
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [all-mpnet-base-v2 Model Card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
