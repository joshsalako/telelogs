# Local Qwen3.6-27B vLLM deployment

This isolated Python 3.12 project serves `Qwen/Qwen3.6-27B` through vLLM's
OpenAI-compatible API for the Telelogs synthesis pipeline. The defaults target
one NVIDIA GPU with no more than 48 GB VRAM: the checkpoint is quantized to four
bits while loading, the context is limited to 16,384 tokens, and the vision
encoder is not loaded.

The upstream checkpoint is about 55.6 GB on disk. Four-bit GPU weights are
smaller, but loading still requires ample system RAM, disk space, and temporary
headroom. Quantization is a memory/quality tradeoff and does not create a new
quantized checkpoint on disk.

## Prerequisites

- Linux on x86-64 with one supported NVIDIA CUDA GPU and current drivers
- Python 3.12, separate from the repository's Python 3.14 environment
- Sufficient RAM and at least 70 GB of free disk/cache space
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Hugging Face access to the model, with a token if required

Check the driver and available memory before installing:

```bash
nvidia-smi
uv --version
```

## Install and configure

From the repository root:

```bash
cd deployment
uv python install 3.12
uv sync --python 3.12
cp .env.example .env
```

`uv sync` installs vLLM 0.19.0 or newer and BitsAndBytes 0.49.2 or newer into
`deployment/.venv`. It does not use the root project's environment. Edit `.env`
to tune the server. Process environment variables override `.env` values.

For gated/authenticated downloads, uncomment `HF_TOKEN` in `.env`. To put the
large cache on another disk, set `HF_HOME` or `HUGGINGFACE_HUB_CACHE`. These are
standard Hugging Face variables inherited by vLLM. Never commit the real `.env`.

To require API authentication, set a strong `QWEN_VLLM_API_KEY`. Export the
same value as `VLLM_API_KEY` when synthesis runs.

## Preview, start, check, and stop

The dry run validates every setting and prints the exact command. It does not
import vLLM, initialize CUDA, contact Hugging Face, or start a server:

```bash
uv run python -m qwen_vllm serve --dry-run
```

Start the server in a dedicated terminal:

```bash
uv run python -m qwen_vllm serve
```

The first real start downloads the original model files into the Hugging Face
cache unless already cached. vLLM performs in-flight BitsAndBytes four-bit
quantization each time it loads the model. Startup can take several minutes. In
a second terminal, verify health, model discovery, and a small chat:

```bash
cd deployment
uv run python -m qwen_vllm check
```

Stop the foreground server with `Ctrl-C` in its terminal. If it is managed by a
service supervisor, use that supervisor's stop command so vLLM exits cleanly.

## Run synthesis

Keep the verified server running. From the repository root, use the separate
root environment:

```bash
cd ..
export VLLM_API_KEY='the-same-value-as-QWEN_VLLM_API_KEY'
uv sync --active
uv run --active python -m synthesis
```

If API protection is disabled, omit the `export`; the synthesis client defaults
to `EMPTY`. The deployment advertises `Qwen3.6-27B`, exactly matching the model
name in `synthesis/config.py`.

## Memory tuning and OOM recovery

The most effective controls in `deployment/.env` are:

- Lower `QWEN_VLLM_MAX_MODEL_LEN` to reduce KV-cache memory. Try `8192`, then
  `4096`. This reduces the prompt-plus-generated-output limit.
- Lower `QWEN_VLLM_MAX_NUM_SEQS` from `8` to `4`, `2`, or `1` to reduce peak
  concurrent KV-cache use and throughput.
- Lower `QWEN_VLLM_GPU_MEMORY_UTILIZATION` from `0.90` to `0.85` if other
  processes or the display need VRAM. Increasing it may help KV-cache capacity,
  but leaves less safety margin.
- Keep `QWEN_VLLM_TENSOR_PARALLEL_SIZE=1` for one GPU and
  `QWEN_VLLM_LANGUAGE_MODEL_ONLY=true`; loading vision consumes more memory.

For an OOM, stop vLLM, use `nvidia-smi` to find other GPU consumers, make one
change at a time, preview it with `serve --dry-run`, and restart. If four-bit
weights alone cannot fit after other GPU processes are removed, use a GPU with
more VRAM or a smaller/pre-quantized model.

## Offline tests

These tests use only the Python standard library and mocked HTTP responses:

```bash
uv run python -m unittest discover -s tests -v
```

They do not import CUDA/vLLM, download weights, or contact a live endpoint.
