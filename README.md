# Telelogs CoT Data Synthesis

An asynchronous multi-agent pipeline for converting 5G network troubleshooting
examples into a randomized, validated Chain-of-Thought supervised fine-tuning
dataset. It targets a local Qwen3.6-27B deployment exposed through vLLM's
OpenAI-compatible chat completions API.

The pipeline creates three independent augmentations per source example,
shuffles table rows without changing measurements, randomizes root-cause labels,
runs elimination- and contradiction-based reasoning agents, validates their
answers with strict majority voting, and formats successful traces into a stable
four-section SFT response.

See [the synthesis documentation](synthesis/README.md) for configuration,
execution, output format, retry behavior, and resume semantics.

## Complete workflow

The model server has its own Linux/CUDA Python 3.12 environment under
[`deployment/`](deployment/README.md). Install it and preview the resolved vLLM
command without downloading the model:

```bash
cd deployment
uv python install 3.12
uv sync --python 3.12
cp .env.example .env
uv run python -m qwen_vllm serve --dry-run
```

After reviewing `.env`, start the server. The first real start downloads the
roughly 55.6 GB upstream checkpoint if it is not already cached, then applies
four-bit BitsAndBytes quantization while loading:

```bash
uv run python -m qwen_vllm serve
```

Keep that terminal open. In another terminal, verify the health endpoint, model
alias, and a small chat completion:

```bash
cd deployment
uv run python -m qwen_vllm check
```

Only after the check passes, run synthesis from the repository root with its
separate Python 3.14 environment:

```bash
cd ..
# If QWEN_VLLM_API_KEY is set in deployment/.env, export the same value here:
# export VLLM_API_KEY='your-key'
uv sync --active
uv run --active python -m synthesis
```

Stop the foreground vLLM server with `Ctrl-C`. See the
[deployment guide](deployment/README.md) for prerequisites, Hugging Face cache
settings, API-key protection, memory tuning, and OOM troubleshooting.

To run all offline tests without contacting vLLM:

```bash
uv run --active python -m unittest discover -s synthesis/tests -v
cd deployment
python3 -m unittest discover -s tests -v
```
