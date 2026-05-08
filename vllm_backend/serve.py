"""
Launch a vLLM OpenAI-compatible inference server.

Example:
    python vllm_backend/serve.py \\
        --model /path/to/Llama-3.1-8B-Instruct \\
        --port 8000 \\
        --tensor-parallel-size 1 \\
        --dtype bfloat16

Then in the main env:
    from consistency import VLLMModel
    provider = VLLMModel(model_name="/path/to/Llama-3.1-8B-Instruct", base_url="http://localhost:8000/v1")
"""
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Start a vLLM OpenAI-compatible server.")
    parser.add_argument("--model", type=str, required=True,
                        help="HF model id or local path to the model weights.")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind the server to.")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind the server to.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism.")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        help="Computation dtype.")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Maximum context length.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="Fraction of GPU memory to reserve for vLLM.")
    parser.add_argument("--api-key", type=str, default="EMPTY",
                        help="Token clients must present; match it on the client side.")
    parser.add_argument("--served-model-name", type=str, default=None,
                        help="Alias exposed to clients (defaults to --model).")
    parser.add_argument("--extra", type=str, nargs=argparse.REMAINDER, default=[],
                        help="Additional flags forwarded verbatim to vllm.entrypoints.openai.api_server.")
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--dtype", args.dtype,
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--api-key", args.api_key,
    ]
    if args.served_model_name:
        cmd += ["--served-model-name", args.served_model_name]
    cmd += args.extra

    print("[vllm_backend] Launching:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
