import modal

vllm_image = modal.Image.debian_slim().pip_install("vllm")

app = modal.App("vllm-test")


@app.function(image=vllm_image, gpu="A10G")
@modal.web_server(8000)
def serve():
    import subprocess

    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "facebook/opt-125m",
        "--port",
        "8000",
    ]
    subprocess.Popen(cmd)
