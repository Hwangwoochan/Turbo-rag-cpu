import argparse
import subprocess
import time
import statistics
import threading

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer


# ------------------------------
# Utility
# ------------------------------

def summarize(arr):
    return {
        "avg": round(statistics.mean(arr), 4),
        "min": round(min(arr), 4),
        "max": round(max(arr), 4),
    }


# ------------------------------
# llama.cpp BENCH
# ------------------------------

def run_llama_once(llama_cli, model, prompt, n_predict, ctx, threads):
    cmd = [
        llama_cli,
        "-m", model,
        "-p", prompt,
        "-n", str(n_predict),
        "-c", str(ctx),
        "-t", str(threads),
        "--temp", "0",
        "--ignore-eos",
        "--no-display-prompt"
    ]

    t0 = time.perf_counter()
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    ttft = None
    assert p.stdout is not None

    while True:
        ch = p.stdout.read(1)
        if ch == "" and p.poll() is not None:
            break

        if ttft is None and ch not in ("", None):
            ttft = time.perf_counter() - t0

        if p.poll() is not None:
            break

    p.wait()
    e2e = time.perf_counter() - t0

    if ttft is None:
        ttft = e2e

    tpot = (e2e - ttft) / max(n_predict - 1, 1)

    return ttft, e2e, tpot


# ------------------------------
# Transformers BENCH
# ------------------------------

@torch.inference_mode()
def run_hf_once(model, tok, prompt, max_new_tokens, threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)

    inputs = tok(prompt, return_tensors="pt")
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        streamer=streamer,
    )

    t0 = time.perf_counter()
    ttft = None
    output_text = []

    def _gen():
        model.generate(**gen_kwargs)

    th = threading.Thread(target=_gen)
    th.start()

    for chunk in streamer:
        if chunk:
            if ttft is None:
                ttft = time.perf_counter() - t0
            output_text.append(chunk)

    th.join()
    e2e = time.perf_counter() - t0

    if ttft is None:
        ttft = e2e

    gen_text = "".join(output_text)
    gen_tokens = len(tok(gen_text, add_special_tokens=False).input_ids)

    if gen_tokens <= 1:
        tpot = 0.0
    else:
        tpot = (e2e - ttft) / (gen_tokens - 1)

    return ttft, e2e, tpot


# ------------------------------
# MAIN
# ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-cli", default="./llama-cli")
    parser.add_argument("--llama-model", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    print("Loading HF model...")
    tok = AutoTokenizer.from_pretrained(args.hf_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=torch.float32,
        device_map=None,
    )
    model.eval()

    # Warmup
    print("Warming up...")
    for _ in range(args.warmup):
        run_llama_once(args.llama_cli, args.llama_model,
                       args.prompt, args.tokens, args.ctx, args.threads)
        run_hf_once(model, tok, args.prompt,
                    args.tokens, args.threads)

    # Bench
    llama_ttft, llama_e2e, llama_tpot = [], [], []
    hf_ttft, hf_e2e, hf_tpot = [], [], []

    print("Benchmarking...")
    for _ in range(args.runs):
        t1 = run_llama_once(args.llama_cli, args.llama_model,
                            args.prompt, args.tokens, args.ctx, args.threads)
        llama_ttft.append(t1[0])
        llama_e2e.append(t1[1])
        llama_tpot.append(t1[2])

        t2 = run_hf_once(model, tok, args.prompt,
                         args.tokens, args.threads)
        hf_ttft.append(t2[0])
        hf_e2e.append(t2[1])
        hf_tpot.append(t2[2])

    print("\n==============================")
    print("CPU BENCHMARK RESULTS")
    print("==============================")

    print("\n--- llama.cpp ---")
    print("Avg E2E :", summarize(llama_e2e))
    print("Avg TTFT:", summarize(llama_ttft))
    print("Avg TPOT:", summarize(llama_tpot))

    print("\n--- Transformers (PyTorch) ---")
    print("Avg E2E :", summarize(hf_e2e))
    print("Avg TTFT:", summarize(hf_ttft))
    print("Avg TPOT:", summarize(hf_tpot))


if __name__ == "__main__":
    main()