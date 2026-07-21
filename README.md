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

## Quick start

Use the uv-managed environment from the repository root:

```bash
uv sync --active
uv run --active python -m synthesis
```

The local vLLM endpoint must already be serving the configured model. To run the
offline tests without contacting vLLM:

```bash
uv run --active python -m unittest discover -s synthesis/tests -v
```
