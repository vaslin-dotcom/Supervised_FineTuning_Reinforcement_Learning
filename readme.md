I am going to create a model to reason like a detective
I have chosen smolLM2 as the pretrained model for this, going to make it undergo a series of Fine Tuning and Reinforcement learning.
I am going to use LoRA for fine tuning

## LoRA Configuration & Dataset Sizing — Calculations

### 1. Model Architecture (SmolLM2-360M)

| Parameter | Value |
|---|---|
| hidden_size | 960 |
| num_hidden_layers | 32 |
| num_attention_heads | 15 |
| num_key_value_heads (GQA) | 5 |
| head_dim | 64 |
| intermediate_size | 2560 |

### 2. Why only `q_proj` and `v_proj`?

LoRA can target any linear projection in the transformer (`q_proj`, `k_proj`, `v_proj`, `o_proj` in attention; `gate_proj`, `up_proj`, `down_proj` in the MLP). We restrict adaptation to **`q_proj` and `v_proj` only**, for two reasons:

- **Task nature — format/style adaptation, not new-knowledge acquisition.** The model already knows the vocabulary and general reasoning ability it needs (from 4T pretraining tokens); what it needs to learn is a *behavioral pattern* — restructuring output into Observation → Possible/Eliminated Scenarios → Inference on mystery-style prompts. This is closer to "change what the model attends to and emphasizes" than "store new facts," which points toward attention-layer adaptation over MLP adaptation (MLP blocks are more associated with factual/knowledge storage in transformer interpretability research).
- **Q and V are the highest-leverage attention components for this.** `q_proj` determines *what a token looks for* in the sequence (i.e., which attention pattern gets formed — directly relevant to learning "attend differently when the input is a mystery-style prompt"). `v_proj` determines *what content gets pulled forward* once attention is placed (relevant to changing output style/content). `k_proj` and `o_proj` contribute comparatively little extra behavioral leverage per the original LoRA paper's ablations, and in SmolLM2's GQA architecture, `k_proj`/`v_proj` are already shared across multiple query heads (5 KV-heads vs. 15 query heads), making `k_proj` a lower-precision lever to adapt. Excluding `k_proj`/`o_proj` also keeps trainable capacity conservative — important given our limited fine-tuning dataset (see below).

### 3. Trainable Parameter Count

LoRA replaces a full weight update to `W` (shape `d_out × d_in`) with two low-rank matrices `A (r × d_in)` and `B (d_out × r)`, giving `r × (d_in + d_out)` trainable parameters per matrix instead of `d_out × d_in`.

| Matrix | d_in | d_out | d_in + d_out |
|---|---|---|---|
| q_proj | 960 | 960 | 1920 |
| v_proj | 960 | 320 | 1280 |
| **Sum (q+v)** | | | **3200** |

Total trainable params = `num_layers × Σ(d_in+d_out) × r` = `32 × 3200 × r`

| Rank (r) | Trainable Params | % of 360M model |
|---|---|---|
| r = 4 | 409,600 | ~0.11% |
| r = 8 | 819,200 | ~0.23% |

Both `r=4` and `r=8` adapters are trained and compared empirically via held-out evaluation (see Evaluation section).

### 4. Dataset Sizing Estimate

**Assumption:** ~350–400 tokens per fully-formatted training example (problem + observations + possible/eliminated scenarios + inference + final answer). We use **375 tokens/example** as the working estimate.

**Guiding heuristic:** total training tokens (across all epochs) should be roughly **10–50×** the trainable parameter count to provide sufficient learning signal without over-memorizing a small dataset. This heuristic is typically calibrated for tasks that teach *new knowledge*; since our task is a lower-complexity *format/style* adaptation on top of an already-capable base model, we treat the **lower end (10–15×)** as a reasonable, defensible target rather than requiring the full 50×.

Required dataset size formula:

N_examples = (ratio × trainable_params) / (tokens_per_example × epochs)

Considering 3 epochs

| Rank | Ratio  | Required Examples |
|---|-------|-------------------|
| r = 4 | 10×   | ~3,000            |
| r = 4 | 50×   | ~14,500           |
| r = 8 | 10×   | ~6,500           |
| r = 8 | 50×   | ~28,500           |

**Decision:** Given free-tier API constraints, we start with a well-designed, diverse dataset of **~800–1,000 examples**, below the theoretical target, and rely on **held-out validation loss + early stopping** (rather than blind epoch count) to empirically catch overfitting during training. Additional *targeted* data is generated later only if error analysis after evaluation identifies specific, recurring failure modes — rather than blindly scaling the dataset upfront.
