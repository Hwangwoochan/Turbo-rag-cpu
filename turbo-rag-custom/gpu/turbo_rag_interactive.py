import argparse
import logging
import sys
import time
from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from qwen2 import Qwen2ModifiedForCausalLM

# LlamaIndex
from llama_index.core import Settings, load_index_from_storage, StorageContext, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

PREFIX = '''<|im_start|>system
You are an accurate and reliable AI assistant that can answer questions with the help of external documents. Please note that external documents may contain noisy information. If the information in the document contains the correct answer, you will give an accurate answer. If the information in the document does not contain the answer, you will generate ’I can not answer the question because of the insufficient information in documents.‘.<|im_end|><|im_start|>user\nDocs:'''


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _step(msg: str, enabled: bool):
    if enabled:
        print(f"[STEP] {msg}")


def _move_past_key_values_to_device(past_key_values, device: torch.device):
    return tuple((k.to(device), v.to(device)) for (k, v) in past_key_values)


def stack_past_key_values(past_key_values_list):
    num_layers = len(past_key_values_list[0])
    batch_past_key_values = []
    for layer in range(num_layers):
        keys = torch.cat([pkv[layer][0] for pkv in past_key_values_list], dim=2)
        values = torch.cat([pkv[layer][1] for pkv in past_key_values_list], dim=2)
        batch_past_key_values.append((keys, values))
    return tuple(batch_past_key_values)


def qa_to_prompt(chunk_list: List[str], query: str) -> str:
    chunk_str = "".join(chunk_list)
    prompt = f'''{PREFIX}{chunk_str}\n\nQuestion: {query}<|im_end|><|im_start|>assistant\n'''
    return prompt


def build_retriever(storage_dir: str, embedding_model_name: str, top_k: int, verbose_steps: bool):
    _step("Building retriever (load index + embedding model)...", verbose_steps)
    Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    index = load_index_from_storage(storage_context)
    retriever = index.as_retriever(similarity_top_k=top_k)
    _step("Retriever ready.", verbose_steps)
    return retriever


def load_model_and_tokenizer(
    model_path: str,
    device: torch.device,
    use_flash_attn: bool,
    use_fast_tokenizer: bool,
    torch_dtype: Optional[str],
    verbose_steps: bool,
):
    _step("Loading model...", verbose_steps)
    attn_implementation = "flash_attention_2" if (use_flash_attn and device.type == "cuda") else None

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map.get(torch_dtype, None)

    # CPU에 fp16/bf16 강제하면 문제나 느려질 수 있어서 fp32로
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32

    model = Qwen2ModifiedForCausalLM.from_pretrained(
        model_path,
        attn_implementation=attn_implementation,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    _step("Model loaded.", verbose_steps)

    _step("Loading tokenizer...", verbose_steps)
    # fast tokenizer가 깨져있는 경우가 있어서 자동 fallback
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast_tokenizer)
    except Exception as e:
        print(f"[WARN] Fast tokenizer load failed: {e}")
        print("[WARN] Falling back to slow tokenizer (use_fast=False)")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _step("Tokenizer ready.", verbose_steps)
    return model, tokenizer


def compute_prefix_kv(model, tokenizer, device: torch.device, verbose_steps: bool):
    _step("Computing PREFIX KV cache...", verbose_steps)
    inputs_prefix = tokenizer([PREFIX], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(
            inputs_prefix["input_ids"].to(device),
            attention_mask=inputs_prefix["attention_mask"].to(device),
            use_cache=True,
        )
    pkv = _move_past_key_values_to_device(out.past_key_values, device)
    _step("PREFIX KV cache ready.", verbose_steps)
    return pkv


def retrieve_chunks_and_optional_kv(
    query_text: str,
    retriever,
    device: torch.device,
    prefix_kv,
    use_kv_cache: bool,
    verbose_steps: bool,
) -> Tuple[List[str], Optional[tuple]]:
    _step(f"Retrieval start (top_k)...", verbose_steps)
    query_bundle = QueryBundle(query_str=query_text)
    retrieved_nodes = retriever.retrieve(query_bundle)
    _step(f"Retrieval done. got {len(retrieved_nodes)} nodes.", verbose_steps)

    chunk_list: List[str] = []
    past_kv = None

    if use_kv_cache:
        _step("Loading chunk KV caches from disk...", verbose_steps)
        kvcache_list = [prefix_kv]
        for nws in retrieved_nodes:
            node = nws.node
            chunk_list.append(node.text)

            kvcache = torch.load(
                node.metadata["kvcache_file_path"],
                weights_only=True,
                map_location=device,
            )
            kvcache = _move_past_key_values_to_device(kvcache, device)
            kvcache_list.append(kvcache)

        past_kv = stack_past_key_values(kvcache_list)
        _step("Chunk KV caches stacked.", verbose_steps)
    else:
        for nws in retrieved_nodes:
            chunk_list.append(nws.node.text)

    return chunk_list, past_kv


@torch.no_grad()
def run_once(
    query_text: str,
    retriever,
    model,
    tokenizer,
    device: torch.device,
    prefix_kv,
    use_kv_cache: bool,
    gen_tokens: int,
    eos_token_ids: List[int],
    verbose_steps: bool,
):
    _sync(device)
    t0 = time.perf_counter()

    chunk_list, past_kv = retrieve_chunks_and_optional_kv(
        query_text=query_text,
        retriever=retriever,
        device=device,
        prefix_kv=prefix_kv,
        use_kv_cache=use_kv_cache,
        verbose_steps=verbose_steps,
    )

    _step("Building prompt...", verbose_steps)
    prompt = qa_to_prompt(chunk_list, query_text)

    _step("Tokenizing prompt...", verbose_steps)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_tokens = int(input_ids.shape[1])

    gen_kwargs = dict(
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        eos_token_id=eos_token_ids,
    )
    if use_kv_cache:
        gen_kwargs["past_key_values"] = past_kv

    # TTFT
    _step("Measuring TTFT (generate 1 token)...", verbose_steps)
    _sync(device)
    t_ttft0 = time.perf_counter()
    _ = model.generate(input_ids, max_new_tokens=1, **gen_kwargs)
    _sync(device)
    ttft = time.perf_counter() - t_ttft0
    _step(f"TTFT done: {ttft:.4f}s", verbose_steps)

    # Full generation
    _step(f"Generating answer (max_new_tokens={gen_tokens})...", verbose_steps)
    _sync(device)
    t_gen0 = time.perf_counter()
    out = model.generate(input_ids, max_new_tokens=gen_tokens, **gen_kwargs)
    _sync(device)
    gen_time = time.perf_counter() - t_gen0
    _step(f"Generation done: {gen_time:.4f}s", verbose_steps)

    total_time = time.perf_counter() - t0

    actual_gen = int(out.shape[1] - input_ids.shape[1])
    if actual_gen <= 1:
        tpot = float("inf")
    else:
        tpot = max(gen_time - ttft, 0.0) / (actual_gen - 1)

    answer_text = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)

    _step("Done (one query).", verbose_steps)

    return {
        "query": query_text,
        "contexts": chunk_list,
        "answer": answer_text.strip(),
        "total_time_s": total_time,
        "ttft_s": ttft,
        "gen_time_s": gen_time,
        "tpot_s_per_token": tpot,
        "input_tokens": input_tokens,
        "gen_tokens": actual_gen,
        "retrieved_chunks": len(chunk_list),
    }


def _print_contexts(contexts: List[str], top_n: int, max_chars: int):
    if not contexts:
        print("[CONTEXTS]\n(none)")
        return

    n = len(contexts) if top_n <= 0 else min(top_n, len(contexts))
    print("[CONTEXTS]")
    for i in range(n):
        ctx = contexts[i].strip()
        if max_chars > 0 and len(ctx) > max_chars:
            ctx = ctx[:max_chars] + "\n... (truncated) ..."
        print(f"\n--- Context {i+1}/{len(contexts)} ---\n{ctx}")


def main():
    parser = argparse.ArgumentParser(description="TurboRAG interactive chat (load once, query many times)")
    parser.add_argument("--model_name", type=str, required=True, help="Local model folder path")
    parser.add_argument("--embedding_model_name", type=str, required=True, help="Embedding model name/path")
    parser.add_argument("--storage_dir", type=str, default="doc_emb", help="Persisted index dir")
    parser.add_argument("--similarity_top_k", type=int, default=20)
    parser.add_argument("--gen_tokens", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--slow_tokenizer", action="store_true")
    parser.add_argument("--torch_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])

    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["kv", "nokv", "both"],
        help="kv: With KV Cache, nokv: Without KV Cache, both: run both each query",
    )

    parser.add_argument("--print_top_n", type=int, default=5, help="How many retrieved contexts to print (<=0 means all)")
    parser.add_argument("--ctx_chars", type=int, default=800, help="Max chars per context to print (<=0 means no truncation)")

    # 🔥 단계 로그
    parser.add_argument("--verbose_steps", action="store_true", help="Print step-by-step progress logs")

    args = parser.parse_args()
    verbose_steps = args.verbose_steps

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        device = torch.device("cpu")

    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_name,
        device=device,
        use_flash_attn=args.use_flash_attn,
        use_fast_tokenizer=not args.slow_tokenizer,
        torch_dtype=args.torch_dtype,
        verbose_steps=verbose_steps,
    )

    retriever = build_retriever(
        storage_dir=args.storage_dir,
        embedding_model_name=args.embedding_model_name,
        top_k=args.similarity_top_k,
        verbose_steps=verbose_steps,
    )

    prefix_kv = compute_prefix_kv(model, tokenizer, device, verbose_steps=verbose_steps)
    eos_token_ids = [151645, 151643]

    print("\n[READY] Model+Retriever loaded.")
    print("Type your question and press Enter.")
    print("Commands: /exit, /quit\n")

    while True:
        try:
            q = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not q:
            continue
        if q in ("/exit", "/quit"):
            print("bye")
            break

        if args.mode == "both":
            modes = [("With KV Cache", True), ("Without KV Cache", False)]
        elif args.mode == "kv":
            modes = [("With KV Cache", True)]
        else:
            modes = [("Without KV Cache", False)]

        for name, use_kv in modes:
            res = run_once(
                query_text=q,
                retriever=retriever,
                model=model,
                tokenizer=tokenizer,
                device=device,
                prefix_kv=prefix_kv,
                use_kv_cache=use_kv,
                gen_tokens=args.gen_tokens,
                eos_token_ids=eos_token_ids,
                verbose_steps=verbose_steps,
            )

            print(f"\n=== {name} ===")
            print(f"[QUERY]\n{res['query']}\n")

            _print_contexts(res["contexts"], top_n=args.print_top_n, max_chars=args.ctx_chars)

            print("\n[ANSWER]")
            print(res["answer"])

            tpot_val = res["tpot_s_per_token"]
            tpot_str = "inf" if tpot_val == float("inf") else f"{tpot_val:.6f}"
            print(
                f"\n[STATS] total={res['total_time_s']:.4f}s | ttft={res['ttft_s']:.4f}s | "
                f"gen={res['gen_time_s']:.4f}s | tpot={tpot_str} | "
                f"in_tok={res['input_tokens']} | gen_tok={res['gen_tokens']} | chunks={res['retrieved_chunks']}"
            )

        print()

if __name__ == "__main__":
    main()