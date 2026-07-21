import modal
import os
import subprocess

MINUTES = 60
HOURS = 60 * MINUTES

# Define the container image with required dependencies
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .pip_install(
        "vllm",
        "huggingface_hub",
        "bitsandbytes",  # Required for bitsandbytes quantization
        "git+https://github.com/huggingface/transformers.git",
    )
    .env({"CACHE_BUST": "3"})
)

app = modal.App("qwen-vllm-deployment")

# Create a volume to cache the Hugging Face model
# This prevents downloading the large model on every container start
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu="A100:1",  # 1x A100 80GB GPU
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    timeout=2 * HOURS,
    secrets=[
        modal.Secret.from_dotenv()
    ],  # This will automatically load HF_TOKEN from your local .env file when deploying
)
@modal.web_server(8000, startup_timeout=15 * MINUTES)
@modal.concurrent(max_inputs=100)
def serve():
    """
    Start the vLLM OpenAI-compatible server.
    We read the configuration from the environment variables (loaded via .env).
    """
    model = os.environ.get("QWEN_VLLM_MODEL", "Qwen/Qwen3.6-27B")
    served_model_name = os.environ.get("QWEN_VLLM_SERVED_MODEL_NAME", "Qwen3.6-27B")

    # We use 0.0.0.0 because Modal routes external traffic to the container on the specified port.
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--served-model-name",
        served_model_name,
        "--tensor-parallel-size",
        os.environ.get("QWEN_VLLM_TENSOR_PARALLEL_SIZE", "1"),
        "--max-model-len",
        os.environ.get("QWEN_VLLM_MAX_MODEL_LEN", "16384"),
        "--gpu-memory-utilization",
        os.environ.get("QWEN_VLLM_GPU_MEMORY_UTILIZATION", "0.90"),
        "--max-num-seqs",
        os.environ.get("QWEN_VLLM_MAX_NUM_SEQS", "8"),
        "--quantization",
        os.environ.get("QWEN_VLLM_QUANTIZATION", "bitsandbytes"),
        "--load-format",
        "bitsandbytes"
        if os.environ.get("QWEN_VLLM_QUANTIZATION", "bitsandbytes") == "bitsandbytes"
        else "auto",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]

    # Optional parameters based on .env
    reasoning_parser = os.environ.get("QWEN_VLLM_REASONING_PARSER")
    if reasoning_parser:
        cmd.extend(["--reasoning-parser", reasoning_parser])

    # Note: vLLM 0.7.3 does not support --language-model-only, so we ignore QWEN_VLLM_LANGUAGE_MODEL_ONLY

    print("Starting vLLM server with command:")
    print(" ".join(cmd))

    # Run the API server subprocess.
    subprocess.Popen(cmd)
