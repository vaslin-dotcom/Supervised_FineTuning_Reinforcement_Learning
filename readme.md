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

### 2. Why `q_proj`, `v_proj`, and `gate_proj`?

LoRA can target any linear projection in the transformer (`q_proj`, `k_proj`, `v_proj`, `o_proj` in attention; `gate_proj`, `up_proj`, `down_proj` in the MLP). We adapt **`q_proj`, `v_proj`, and `gate_proj`**, for the following reasons:

- **Task nature — format/style adaptation, not new-knowledge acquisition, with added capacity for output shaping.** The model already knows the vocabulary and general reasoning ability it needs (from 4T pretraining tokens); what it needs to learn is a *behavioral pattern* — restructuring output into Observation → Possible/Eliminated Scenarios → Inference on mystery-style prompts. This points toward attention-layer adaptation over full MLP adaptation.
- **Q and V are the highest-leverage attention components for this.** `q_proj` determines *what a token looks for* in the sequence (i.e., which attention pattern gets formed — directly relevant to learning "attend differently when the input is a mystery-style prompt"). `v_proj` determines *what content gets pulled forward* once attention is placed (relevant to changing output style/content). `k_proj` and `o_proj` contribute comparatively little extra behavioral leverage per the original LoRA paper's ablations, and in SmolLM2's GQA architecture, `k_proj`/`v_proj` are already shared across multiple query heads (5 KV-heads vs. 15 query heads), making `k_proj` a lower-precision lever to adapt.
- **`gate_proj` is included for additional output-shaping capacity.** `gate_proj` sits in SmolLM2's gated MLP block and contributes to how strongly different learned features are expressed in the model's output. Including it gives the adapter more capacity to shift generation style/structure beyond what attention-only adaptation provides, which is useful given the low LoRA rank and modest dataset size used here. `up_proj`/`down_proj` are excluded to keep trainable capacity conservative.

### 3. Trainable Parameter Count

LoRA replaces a full weight update to `W` (shape `d_out × d_in`) with two low-rank matrices `A (r × d_in)` and `B (d_out × r)`, giving `r × (d_in + d_out)` trainable parameters per matrix instead of `d_out × d_in`.

| Matrix | d_in | d_out | d_in + d_out |
|---|---|---|---|
| q_proj | 960 | 960 | 1920 |
| v_proj | 960 | 320 | 1280 |
| gate_proj | 960 | 2560 | 3520 |
| **Sum (q+v+gate)** | | | **6720** |

Total trainable params = `num_layers × Σ(d_in+d_out) × r` = `32 × 6720 × r`

| Rank (r) | Trainable Params | % of 360M model |
|---|---|---|
| r = 4 | 860,160 | ~0.24% |
| r = 8 | 1,720,320 | ~0.48% |

Both `r=4` and `r=8` adapters were trained and compared. **r=8 was selected** going forward, as r=4 did not perform well enough on the reasoning-structure task on qualitative review of generations.

### 4. Dataset Sizing Estimate

**Assumption:** ~350–400 tokens per fully-formatted training example (problem + observations + possible/eliminated scenarios + inference + final answer). We use **375 tokens/example** as the working estimate.

**Guiding heuristic:** total training tokens (across all epochs) should be roughly **10–50×** the trainable parameter count to provide sufficient learning signal without over-memorizing a small dataset. This heuristic is typically calibrated for tasks that teach *new knowledge*; since our task is a lower-complexity *format/style* adaptation on top of an already-capable base model, we treat the **lower end (10–15×)** as a reasonable, defensible target rather than requiring the full 50×.

Required dataset size formula:

N_examples = (ratio × trainable_params) / (tokens_per_example × epochs)

Training is run for **10 epochs**.

| Rank | Ratio | Required Examples (10 epochs) |
|---|---|---|
| r = 4 | 10× | ~2,300 |
| r = 4 | 15× | ~3,440 |
| r = 8 | 10× | ~4,590 |
| r = 8 | 15× | ~6,880 |

**Actual dataset used:** 856 training examples, 120 validation examples. This sits below the 10× target for r=8 at 10 epochs, so held-out validation loss is monitored closely during training to catch overfitting given the dataset is smaller than the heuristic target for the chosen rank and target-module set.

---

## RL Phase

*(To be added later — reward function design, GRPO setup, and results will be documented in a separate section once finalized.)*
