# 🕵️ Detective LLM: Teaching a Small Model to Reason Like a Detective

People kept asking me, "So AI engineering is just prompting, right?" This project is my answer. I fine-tuned a small language model (**SmolLM2-360M**) with **LoRA**, then tried to improve its reasoning with **reinforcement learning (GRPO)**. Given a short statement, the model responds in a fixed detective-style format.

This is a learning project, not a production model. The negative results are documented on purpose.

## Example

**Input**

```
<<STATEMENT>>
Maria left home at 8 a.m. with an umbrella even though the sky was clear. When she came back in the evening, her clothes were soaked.
```

**Output format**

```
<<OBSERVATIONS>>     what the statement actually says
<<POSSIBILITIES>>    explanations consistent with the facts
<<IMPOSSIBILITIES>>  explanations the facts rule out
<<DEDUCTION>>        reasoning from the above
<<FINAL ANSWER>>     the conclusion
```

> Add a screenshot of a real generation here: `![example](assets/example.png)`

## Pipeline

```
Synthetic dataset  →  SFT with LoRA  →  RL (GRPO)  →  Evaluation & error analysis
   (976 examples)     (format learned)   (reward design)      (in progress)
```

## Base model

**SmolLM2-360M**: hidden size 960, 32 layers, 15 attention heads, 5 KV heads (GQA), head dim 64, MLP size 2560.

## 1. Dataset

- 986 examples generated with an LLM, filtered to **976** by removing examples over 1,222 tokens.
- Each example has a `context` (crime/mystery, everyday logic, scientific reasoning, pure facts), a `difficulty` (1-5) and a `deduction`.
- Formatted as `prompt` (`<<STATEMENT>>` + problem) and `target` (the five marker sections). An EOS token is appended to the target, and the prompt tokens are masked in the loss.
- Split: **856 train / 120 validation** (random, seed 1).
- **Caveat:** the targets are LLM-generated and not yet manually audited. The dataset is also far smaller than my own sizing estimate (see the calculations below).

## 2. Supervised fine-tuning (LoRA)

Shared settings: lr 2e-5, 10 epochs, batch size 10, LoRA dropout 0.05, fp16, early stopping (patience 2), best model chosen by validation loss.

| Config | Target modules | Trainable params | Val loss | Token acc. | Result |
|---|---|---|---|---|---|
| r=4, α=8 | q, v | 409,600 | 1.628 | 0.632 | Rarely followed the format |
| r=8, α=16 | q, v | 819,200 | 1.545 | 0.646 | Better loss, format still unreliable |
| r=4, α=8 | q, v, gate_proj | 860,160 | _TODO_ | _TODO_ | **Format reliably learned** |

### What I learned

- I first restricted LoRA to `q_proj` and `v_proj`, reasoning that this task is about behavior and style rather than new knowledge. That turned out to be too little capacity for switching output format.
- Adding **only `gate_proj`** (a one-variable change) made the model reliably produce all five sections in order, under greedy and sampled decoding.
- Higher repetition penalty (1.3) produced junk on long generations. Greedy decoding with the q/v-only models looped.
- **The remaining problem is content, not format.** The model invents details that are not in the statement and sometimes contradicts itself in the final answer.

## 3. Reinforcement learning (GRPO)

Starting point: the q/v + gate_proj checkpoint. Settings: group size G=4, 10 prompts per step, KL beta 0.05. RL prompts come from the 856 training examples, and evaluation uses the held-out 120.

### Roadblocks

1. **Reward hacking with a rule-based reward.** The model exploited the reward. In one pilot, a completion with circular reasoning and junk text after the final answer scored the highest reward in the batch. I stopped the run and fixed the reward.
2. **LLM-as-judge on free APIs.** RL needs thousands of judge calls, and the rate limits made this unworkable.
3. **Paid API credits ($5).** Still not enough for RL-scale judging.
4. **Jev as a reward model.** [Jev](https://www.typesafe.ai) (TypeSafe AI) is a non-autoregressive model that returns probability distributions over typed answers instead of generating text. _TODO: what I asked it to judge, how I converted its probabilities into a reward, and what happened._

### Rule-based reward design

The reward is **multiplicative**, so one fatal flaw sinks the whole score:

```
reward = structure × groundedness × duplication × contradiction-free × distinctness
```

- **structure:** fraction of the 5 markers present. Hard-fails on stray markers, duplicate markers or wrong order.
- **groundedness / contradiction:** sentence-embedding similarity (all-MiniLM-L6-v2, CPU only).
- **duplication / distinctness:** penalizes copied text between sections, and possibilities identical to impossibilities.

The similarity thresholds still need calibration against labeled data.

### Result

_TODO: add the outcome of the Jev-reward run (reward curve, or before/after generations). The model did not improve enough._

## Findings

- Format adherence and reasoning quality are separate problems. LoRA on the right modules solved the first, not the second.
- A reward function will be exploited in ways you did not anticipate. Reading raw completions by hand caught what the reward numbers hid.
- Scoring outputs at RL scale is limited by cost and rate limits, and the reward design matters at least as much as the algorithm.
- A 360M model may not have the capacity for grounded multi-step reasoning.

## Roadmap

- [ ] Manually audit 15-20 training targets for groundedness
- [ ] Build the evaluation suite (format adherence, groundedness, reasoning quality, sliced by context and difficulty)
- [ ] Repeat the pipeline with a stronger base model
- [ ] Calibrate the embedding-based reward against labeled data

## LoRA sizing calculations

Trainable parameters = layers × Σ(d_in + d_out) × r, with q_proj (960→960), v_proj (960→320) and gate_proj (960→2560).

| Modules | Σ(d_in + d_out) | r=4 | r=8 |
|---|---|---|---|
| q, v | 3,200 | 409,600 | 819,200 |
| q, v, gate | 6,720 | 860,160 | 1,720,320 |

Dataset sizing heuristic: total training tokens should be roughly 10-50× the trainable parameters. With ~375 tokens per example, that suggests ~8,600 examples for r=4 at 10×, and I used 976. I relied on validation loss and early stopping to catch overfitting.

## Repository structure

| File | Purpose |
|---|---|
| `llm.py` | _TODO_ |
| `main.py` | _TODO_ |
| `schemas.py` | _TODO_ |
| `notebooks/` | _TODO: SFT, RL and generation notebooks_ |

## Setup

_TODO: Python version, `pip install -r requirements.txt`, and where to put API keys as environment variables (never commit keys)._

Compute: Colab Pro and Kaggle GPUs (free tier for RL).

## Acknowledgements

Base model: [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) by Hugging Face.
