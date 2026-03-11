import argparse
import json
import os
import time
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import ollama
from sentence_transformers import SentenceTransformer


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def make_prompt(contexts: List[Tuple[str, str, float]], query: str) -> str:
    ctx_block = []
    for i, (src, chunk, score) in enumerate(contexts, 1):
        ctx_block.append(f"[{i}] (score={score:.4f}) source={src}\n{chunk}")
    ctx = "\n\n".join(ctx_block)

    prompt = f"""You are a helpful assistant. Use ONLY the information in the CONTEXT to answer.
If the answer is not in the context, say "I don't know based on the provided context."

CONTEXT:
{ctx}

QUESTION:
{query}

ANSWER:
"""
    return prompt


def retrieve_numpy(
    query: str,
    embedder: SentenceTransformer,
    embeddings: np.ndarray,            # [N, D], normalized
    chunks: List[str],
    metas: List[Dict[str, Any]],
    top_k: int,
) -> Tuple[List[Tuple[str, str, float]], float]:
    """
    Returns (contexts, retr_time_s).
    retr_time includes query embedding + scoring + top-k selection.
    """
    t0 = time.perf_counter()
    qv = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")[0]  # [D]
    scores = embeddings @ qv  # [N]

    k = min(top_k, scores.shape[0])
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    out: List[Tuple[str, str, float]] = []
    for idx in top_idx.tolist():
        meta = metas[idx]
        src = os.path.basename(meta.get("doc_path", "unknown"))
        out.append((src, chunks[idx], float(scores[idx])))

    retr_time = time.perf_counter() - t0
    return out, retr_time


def ollama_stream_with_ttft_tpot(
    model: str,
    prompt: str,
    num_ctx: int,
    num_predict: int,
) -> Tuple[str, float, float, int, float]:
    """
    Returns:
      text, ttft_s, gen_time_s, gen_tokens, tpot_s_per_token
    TTFT: time until first content chunk arrives.
    TPOT: (gen_time - ttft) / (gen_tokens - 1) if possible else gen_time/gen_tokens.
    """
    t_start = time.perf_counter()
    first_token_time: Optional[float] = None

    parts: List[str] = []
    last_meta: Dict[str, Any] = {}

    stream = ollama.generate(
        model=model,
        prompt=prompt,
        stream=True,
        options={
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": 0.0,
        },
    )

    for chunk in stream:
        # chunk 예시: {"response": "...", "done": False, ...}
        s = chunk.get("response", "")
        if s and first_token_time is None:
            first_token_time = time.perf_counter()
        if s:
            parts.append(s)
        last_meta = chunk  # done=True 마지막 chunk가 메타 포함하는 경우가 많음

    t_end = time.perf_counter()

    text = "".join(parts)
    gen_time = t_end - t_start
    ttft = (first_token_time - t_start) if first_token_time is not None else gen_time

    # Ollama가 주는 토큰 카운트 (버전에 따라 eval_count가 있을 때가 많음)
    gen_tokens = 0
    if isinstance(last_meta, dict):
        gen_tokens = int(last_meta.get("eval_count", 0) or 0)

    # TPOT 계산
    if gen_tokens >= 2:
        tpot = max(gen_time - ttft, 0.0) / (gen_tokens - 1)
    elif gen_tokens == 1:
        tpot = float("inf")
    else:
        # eval_count가 없으면 fallback: 대략(공백 기준) — 정확하진 않음
        approx = max(len(text.split()), 1)
        gen_tokens = 0  # "정확한 token"이 아니므로 0으로 표시
        tpot = gen_time / approx

    return text.strip(), ttft, gen_time, gen_tokens, tpot


def main():
    parser = argparse.ArgumentParser(description="Naive RAG (numpy cosine) + Ollama with TTFT/TPOT stats")
    parser.add_argument("--ollama_model", type=str, required=True)
    parser.add_argument("--index_dir", type=str, required=True, help="embeddings.npy/meta.jsonl/chunks.jsonl")
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_ctx", type=int, default=4096)
    parser.add_argument("--num_predict", type=int, default=256)
    parser.add_argument("--print_ctx_chars", type=int, default=600)
    args = parser.parse_args()

    emb_path = os.path.join(args.index_dir, "embeddings.npy")
    meta_path = os.path.join(args.index_dir, "meta.jsonl")
    chunks_path = os.path.join(args.index_dir, "chunks.jsonl")

    print("[STEP] Loading embeddings.npy + meta/chunks...")
    embeddings = np.load(emb_path).astype("float32")
    metas = load_jsonl(meta_path)
    chunks_rows = load_jsonl(chunks_path)
    chunks = [r["text"] for r in chunks_rows]

    if not (len(metas) == len(chunks) == embeddings.shape[0]):
        print(f"[WARN] size mismatch: metas={len(metas)}, chunks={len(chunks)}, embeddings={embeddings.shape[0]}")

    # normalize once if needed
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if not np.allclose(norms.mean(), 1.0, atol=1e-2):
        print("[WARN] embeddings seem not normalized; normalizing now...")
        embeddings = embeddings / (norms + 1e-12)

    print("[STEP] Loading embedding model (for queries only)...")
    embedder = SentenceTransformer(args.embed_model)

    print("\n[READY] Loaded numpy index. Type a question. Commands: /exit, /quit\n")

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

        # 1) retrieval
        ctxs, retr_s = retrieve_numpy(q, embedder, embeddings, chunks, metas, top_k=args.top_k)

        print("\n[QUERY]")
        print(q)

        print("\n[CONTEXTS]")
        for i, (src, chunk, score) in enumerate(ctxs, 1):
            ch = chunk.strip()
            if args.print_ctx_chars > 0 and len(ch) > args.print_ctx_chars:
                ch = ch[:args.print_ctx_chars] + "\n... (truncated) ..."
            print(f"\n--- {i}/{len(ctxs)} score={score:.4f} source={src} ---\n{ch}")

        # 2) prompt + generation (TTFT/TPOT)
        prompt = make_prompt(ctxs, q)
        ans, ttft_s, gen_s, gen_tokens, tpot = ollama_stream_with_ttft_tpot(
            model=args.ollama_model,
            prompt=prompt,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
        )

        print("\n[ANSWER]")
        print(ans)

        tpot_str = "inf" if tpot == float("inf") else f"{tpot:.6f}"
        tok_str = f"{gen_tokens}" if gen_tokens > 0 else "unknown"

        print(
            f"\n[STATS] retr={retr_s:.4f}s | ttft={ttft_s:.4f}s | gen={gen_s:.4f}s | "
            f"tpot={tpot_str} | gen_tok={tok_str} | top_k={args.top_k} | index_chunks={embeddings.shape[0]}\n"
        )


if __name__ == "__main__":
    main()