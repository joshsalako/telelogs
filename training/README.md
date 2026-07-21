# Two-stage RCA alignment

This directory is an independent Python 3.12 project for aligning
`Qwen/Qwen3.5-4B` on a single NVIDIA GPU with up to 48 GB VRAM. Stage one uses
the synthesized reasoning trajectories. Stage two returns to the original
2,400-question distribution, samples eight new trajectories per question, and
rewards only exact agreement with the original `C1`-`C8` label. The 864
validation questions are never used for training.

The saved `SFT_Model` and `RL_Model` directories are PEFT LoRA adapters, not
merged model copies. `RL_Model` contains the cumulative SFT-plus-GRPO update and
loads directly over the original base model.

## Requirements and installation

- Linux with Python 3.12, an NVIDIA GPU, a working CUDA toolkit, and a recent
  NVIDIA driver.
- Enough local storage for the base model, optimizer state, adapters, and
  checkpoints.
- Flash Attention 2 needs a CUDA-compatible PyTorch build, the CUDA compiler,
  a C++ compiler, and `ninja`. Install PyTorch before building `flash-attn` if
  uv cannot build both in one transaction.

From this directory:

```bash
cd training
uv python install 3.12
uv sync --extra flash
```

If the isolated Flash Attention build cannot see PyTorch, use the documented
two-pass uv approach:

```bash
uv sync --no-install-package flash-attn
uv sync --extra flash --no-build-isolation-package flash-attn
```

Add `--extra wandb` to either sync command to install W&B. No network access or
model download occurs during the offline tests or structural data validation.
Model weights are downloaded only when a training/evaluation command first
loads the configured base model.

## Data validation

Run from `training/` after synthesis has produced
`../synthesis/sft_train_data.jsonl`:

```bash
uv run python -m rca_training validate-data
```

The command rejects malformed JSON/CSV, empty or duplicate records, invalid
labels, synthetic fields in `train.json`, broken validation coverage, and
anything other than four identical validation targets per base ID. During
development, the immutable original and validation inputs can be checked before
synthesis finishes:

```bash
uv run python -m rca_training validate-data --skip-sft
```

SFT sequence length and GRPO prompt length are additionally checked with the
actual model tokenizer when those stages run; records are rejected rather than
silently truncated.

## Commands

Stage-one supervised fine-tuning uses completion-only cross-entropy for 10
epochs, BF16, micro-batch 1, 128 gradient accumulation steps, and an 8,192-token
combined limit:

```bash
uv run python -m rca_training sft
```

Stage-two GRPO loads `artifacts/SFT_Model`, samples eight responses at
temperature 0.7/top-p 0.95, computes exact binary rewards from the final
`\boxed{...}`, normalizes advantages inside each eight-response group, and uses
a frozen copy of the initial SFT adapter for the KL term:

```bash
uv run python -m rca_training grpo
```

Evaluation loads the base model plus `artifacts/RL_Model`, samples four times
per validation question, and writes `metrics.json`, `samples.jsonl`, and
`questions.jsonl` below `artifacts/evaluation/`:

```bash
uv run python -m rca_training evaluate
```

`pass@1` is the number of correct samples divided by `864 * 4`. `maj@4` requires
a unique plurality; a 2-2 tie is incorrect. Missing, malformed, differently
cased, or whitespace-padded boxed contents are incorrect.

Every GPU command supports `--help`. Common overrides include `--model`,
`--artifact-root`, `--seed`, `--attention-implementation sdpa`, and
`--wandb-project NAME`. W&B is disabled unless the project flag is supplied;
local log files under `artifacts/logs/` are always written.

## Checkpoints and resume

Both stages write adapter-only `checkpoint-N` directories at optimizer-step
boundaries. Each checkpoint includes the LoRA adapter, optimizer state, and a
small JSON cursor. Resume the matching stage explicitly:

```bash
uv run python -m rca_training sft --resume-from-checkpoint artifacts/SFT_Model/checkpoint-100
uv run python -m rca_training grpo --resume-from-checkpoint artifacts/RL_Model/checkpoint-100
```

SFT resumes at the next micro-batch. GRPO resumes at the next original training
record and reconstructs its KL reference from the unchanged final SFT adapter.
Do not point one stage at the other stage's checkpoint or change the dataset,
seed, batch geometry, or model while resuming.

## Memory tuning and OOM recovery

The defaults are intended for one 48 GB device. Flash Attention 2, BF16,
gradient checkpointing, and language-only LoRA keep the largest allocations
bounded; no second base-model copy or colocated vLLM server is created.

If CUDA runs out of memory:

1. Resume from the latest completed checkpoint; do not reuse the interrupted
   partial group.
2. Reduce `--completion-max-length` for GRPO/evaluation, or use a shorter
   explicitly approved `--prompt-max-length` if all records still fit.
3. Use `--attention-implementation sdpa` only when Flash Attention is
   unavailable; SDPA can use more memory.
4. Close other GPU processes and make sure no vLLM server is occupying the
   device. `nvidia-smi` should show the expected free memory before restart.

Do not reduce GRPO `--num-generations` without setting the same
`--gradient-accumulation-steps`: a complete response group must form every
optimizer update. Reducing SFT gradient accumulation changes its effective
batch size from the specified 128.

## Offline verification

The tests use only the standard library and never load model weights:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
