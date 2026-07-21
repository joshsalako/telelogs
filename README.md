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

The vLLM model server is deployed using Modal. To review and edit the Modal deployment script:

```bash
cat modal_qwen_deploy.py
```

To run the vLLM server on Modal ephemerally (e.g. for development or testing):

```bash
uv run modal serve modal_qwen_deploy.py
```

To permanently deploy the model server to Modal so that it runs persistently:

```bash
uv run modal deploy modal_qwen_deploy.py
```

To take down the permanent deployment and stop incurring costs when you are finished:

```bash
uv run modal app stop qwen-vllm-deployment
```

After deploying to Modal, verify that the health endpoint and chat completion respond successfully using Python or `curl`.

Only after the server is deployed and responding, run the synthesis from the repository root with its Python 3.14 environment:

```bash
uv sync --active
uv run --active python -m synthesis
```

See the [synthesis documentation](synthesis/README.md) for configuring the dataset generation pipeline, model URL endpoints, retry behavior, and checkpointing.

To run all offline tests without contacting vLLM:

```bash
uv run --active python -m unittest discover -s synthesis/tests -v
cd deployment
uv run python -m unittest discover -s tests -v
```
