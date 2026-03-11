# turbo_rag_bench.py
import argparse
import json
import logging
import sys
import time
from typing import List, Optional, Tuple

import torch
from tabulate import tabulate
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


def _move_past_key_values_to_device(past_key_values, device: torch.device):
    """
    past_key_values: tuple of layers, each layer is (k, v) tensors
    """
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
    prompt = f'''{PREFIX}{chunk_str}\n\nQuestuin: {query}<|im_end|><|im_start|>assistant\n'''
    return prompt


def load_questions(path: str, n: int) -> List[str]:
    qs = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            qs.append(d["query"])
            if len(qs) >= n:
                break
    return qs


def build_retriever(storage_dir: str, embedding_model_name: str, top_k: int):
    Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    index = load_index_from_storage(storage_context)
    return index.as_retriever(similarity_top_k=top_k)


def load_model_and_tokenizer(
    model_path: str,
    device: torch.device,
    use_flash_attn: bool,
    use_fast_tokenizer: bool,
    torch_dtype: Optional[str],
):
    attn_implementation = "flash_attention_2" if (use_flash_attn and device.type == "cuda") else None

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(torch_dtype, None)

    # CPU에 fp16/bf16 강제하면 느리거나 문제날 수 있어서 기본 fp32 권장
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32

    model = Qwen2ModifiedForCausalLM.from_pretrained(
        model_path,
        attn_implementation=attn_implementation,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast_tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def compute_prefix_kv(model, tokenizer, device: torch.device):
    inputs_prefix = tokenizer([PREFIX], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(
            inputs_prefix["input_ids"].to(device),
            attention_mask=inputs_prefix["attention_mask"].to(device),
            use_cache=True,
        )
    # 혹시 모르게 device 통일
    return _move_past_key_values_to_device(out.past_key_values, device)


def measure_one_query(
    query_text: str,
    retriever,
    model,
    tokenizer,
    device: torch.device,
    prefix_kv,
    use_chunk_cache: bool,
    gen_tokens: int,
    eos_token_ids: List[int],
) -> Tuple[float, float, float, int, int]:
    """
    Returns:
      e2e_time: retrieval + (optional KV load) + prompt build + generate(gen_tokens) 포함 시간
      ttft: generate(1 token) 시간 (모델 생성 구간 기준)
      tpot: (generate(gen_tokens) - ttft) / (gen_tokens-1)
      actual_gen_tokens: 실제 생성된 토큰 수
      input_tokens: 입력 토큰 수(=context + query 포함 프롬프트 토큰 길이)
    """

    _sync(device)
    e2e_t0 = time.perf_counter()

    # retrieval
    query_bundle = QueryBundle(query_str=query_text)
    retrieved_nodes = retriever.retrieve(query_bundle)

    # KV + chunks
    kvcache_list, chunk_list = [prefix_kv], []

    if use_chunk_cache:
        for nws in retrieved_nodes:
            node = nws.node

            # CPU 실험이면 KV-cache 파일에 CUDA 텐서가 들어있어도 강제로 CPU로 로드
            kvcache = torch.load(
                node.metadata["kvcache_file_path"],
                weights_only=True,
                map_location=device,
            )
            # 내부 텐서도 확실히 device로
            kvcache = _move_past_key_values_to_device(kvcache, device)

            kvcache_list.append(kvcache)
            chunk_list.append(node.text)

        past_kv = stack_past_key_values(kvcache_list)
    else:
        for nws in retrieved_nodes:
            chunk_list.append(nws.node.text)
        past_kv = None

    # prompt + encode
    prompt = qa_to_prompt(chunk_list, query_text)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_tokens = int(input_ids.shape[1])  # 🔥 입력 토큰 수 출력용

    # common gen kwargs
    gen_kwargs = dict(
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        eos_token_id=eos_token_ids,
    )
    if use_chunk_cache:
        gen_kwargs["past_key_values"] = past_kv

    # TTFT (generate 1 token)
    _sync(device)
    t0 = time.perf_counter()
    _ = model.generate(input_ids, max_new_tokens=1, **gen_kwargs)
    _sync(device)
    ttft = time.perf_counter() - t0

    # Total generation time for gen_tokens
    _sync(device)
    t1 = time.perf_counter()
    out = model.generate(input_ids, max_new_tokens=gen_tokens, **gen_kwargs)
    _sync(device)
    total_gen_time = time.perf_counter() - t1

    actual_gen = int(out.shape[1] - input_ids.shape[1])
    if actual_gen <= 1:
        tpot = float("inf")
    else:
        tpot = max(total_gen_time - ttft, 0.0) / (actual_gen - 1)

    _sync(device)
    e2e_time = time.perf_counter() - e2e_t0

    return e2e_time, ttft, tpot, actual_gen, input_tokens


def bench_device(device_str: str, args) -> List[List[str]]:
    device = torch.device(device_str)

    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_name,
        device=device,
        use_flash_attn=args.use_flash_attn,
        use_fast_tokenizer=not args.slow_tokenizer,
        torch_dtype=args.torch_dtype,
    )

    retriever = build_retriever(
        storage_dir=args.storage_dir,
        embedding_model_name=args.embedding_model_name,
        top_k=args.similarity_top_k,
    )

    prefix_kv = compute_prefix_kv(model, tokenizer, device)

    questions = load_questions(args.query_file, args.num_questions)

    rows = []
    eos_token_ids = [151645, 151643]

    for mode_name, use_cache in [("With KV Cache", True), ("Without KV Cache", False)]:
        e2e_times: List[float] = []
        ttfts: List[float] = []
        tpots: List[float] = []
        gens: List[int] = []
        in_tokens_list: List[int] = []

        for q in questions:
            e2e, ttft, tpot, g, in_tokens = measure_one_query(
                query_text=q,
                retriever=retriever,
                model=model,
                tokenizer=tokenizer,
                device=device,
                prefix_kv=prefix_kv,
                use_chunk_cache=use_cache,
                gen_tokens=args.gen_tokens,
                eos_token_ids=eos_token_ids,
            )
            e2e_times.append(e2e)
            ttfts.append(ttft)
            tpots.append(tpot)
            gens.append(g)
            in_tokens_list.append(in_tokens)

            # 쿼리별 토큰 수를 보고 싶으면 주석 해제
            # print(f"[TOKENS] mode={mode_name} input_tokens={in_tokens} gen_tokens={g}")

        avg_e2e = sum(e2e_times) / len(e2e_times)
        avg_ttft = sum(ttfts) / len(ttfts)

        finite_tpots = [x for x in tpots if x != float("inf")]
        avg_tpot = (sum(finite_tpots) / len(finite_tpots)) if finite_tpots else float("inf")
        avg_gen = sum(gens) / len(gens)
        avg_in_tokens = sum(in_tokens_list) / len(in_tokens_list)

        rows.append([
            device_str,
            mode_name,
            f"{avg_e2e:.6f}",
            f"{avg_ttft:.6f}",
            ("inf" if avg_tpot == float("inf") else f"{avg_tpot:.6f}"),
            f"{avg_gen:.1f}",
            f"{avg_in_tokens:.1f}",  # 🔥 평균 입력 토큰
        ])

    return rows


def main():
    parser = argparse.ArgumentParser(description="TurboRAG benchmark: E2E avg time + TTFT + TPOT on GPU/CPU")

    parser.add_argument("--model_name", type=str, required=True, help="Local model folder path")
    parser.add_argument("--embedding_model_name", type=str, required=True, help="Embedding model name/path")
    parser.add_argument("--storage_dir", type=str, default="doc_emb", help="Persisted index dir (doc_emb)")
    parser.add_argument("--query_file", type=str, default="./questions/query.jsonl", help="Queries jsonl")
    parser.add_argument("--num_questions", type=int, default=50)
    parser.add_argument("--similarity_top_k", type=int, default=20)

    parser.add_argument("--gen_tokens", type=int, default=32, help="Tokens to generate for TPOT measurement")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "both"])
    parser.add_argument("--use_flash_attn", action="store_true", help="Use FlashAttention2 (CUDA only)")
    parser.add_argument("--slow_tokenizer", action="store_true", help="Force use_fast=False")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Model dtype (CPU will fallback to fp32 if fp16/bf16)",
    )

    args = parser.parse_args()

    all_rows: List[List[str]] = []
    headers = [
        "Device",
        "Mode",
        "Avg E2E Time (s)",
        "Avg TTFT (s)",
        "Avg TPOT (s/token)",
        "Avg Gen Tokens",
        "Avg Input Tokens",  # 🔥 추가
    ]

    if args.device == "both":
        if torch.cuda.is_available():
            all_rows.extend(bench_device("cuda", args))
        all_rows.extend(bench_device("cpu", args))
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available; falling back to CPU.")
            all_rows.extend(bench_device("cpu", args))
        else:
            all_rows.extend(bench_device(args.device, args))

    print(tabulate(all_rows, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()