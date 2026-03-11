import argparse
import glob
import json
import os
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    """
    Very naive chunking by characters.
    Returns list of dicts:
      {"text": chunk_text, "char_start": i, "char_end": j}
    """
    text = text.replace("\r\n", "\n")
    n = len(text)
    step = max(chunk_size - overlap, 1)

    chunks: List[Dict[str, Any]] = []
    i = 0
    while i < n:
        j = min(i + chunk_size, n)
        ch = text[i:j].strip()
        if ch:
            chunks.append({"text": ch, "char_start": i, "char_end": j})
        i += step
    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Build naive RAG index (numpy cosine) - save embeddings.npy + chunks/meta jsonl"
    )
    parser.add_argument("--docs_dir", type=str, required=True, help="Directory containing .txt files (recursive)")
    parser.add_argument("--out_dir", type=str, default="./rag_index_minilm_np", help="Output directory")
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--chunk_size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # recursive txt scan
    paths = sorted(glob.glob(os.path.join(args.docs_dir, "**/*.txt"), recursive=True))
    if not paths:
        raise SystemExit(f"No .txt files found in: {args.docs_dir}")

    print(f"[STEP] Found {len(paths)} txt files under {args.docs_dir}")
    print(f"[STEP] Loading embedding model: {args.embed_model}")
    embedder = SentenceTransformer(args.embed_model)

    chunk_texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    print("[STEP] Chunking documents...")
    for p in paths:
        text = read_txt(p)
        chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
        for ci, ch in enumerate(chunks):
            chunk_texts.append(ch["text"])
            metas.append(
                {
                    "doc_path": p,
                    "chunk_id": ci,
                    "char_start": ch["char_start"],
                    "char_end": ch["char_end"],
                }
            )

    if not chunk_texts:
        raise SystemExit("No chunks produced. Check your documents encoding/content.")

    print(f"[STEP] Total chunks: {len(chunk_texts)}")
    print("[STEP] Embedding chunks (normalize_embeddings=True)...")
    embs = embedder.encode(
        chunk_texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,  # 중요: cosine용
        show_progress_bar=True,
    ).astype("float32")

    # sanity
    if embs.ndim != 2:
        raise SystemExit(f"Unexpected embedding shape: {embs.shape}")

    out_emb = os.path.join(args.out_dir, "embeddings.npy")
    out_chunks = os.path.join(args.out_dir, "chunks.jsonl")
    out_meta = os.path.join(args.out_dir, "meta.jsonl")
    out_info = os.path.join(args.out_dir, "info.json")

    print("[STEP] Saving embeddings.npy ...")
    np.save(out_emb, embs)

    print("[STEP] Saving chunks.jsonl + meta.jsonl ...")
    with open(out_chunks, "w", encoding="utf-8") as f:
        for t in chunk_texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    with open(out_meta, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    info = {
        "embed_model": args.embed_model,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "batch_size": args.batch_size,
        "num_docs": len(paths),
        "num_chunks": len(chunk_texts),
        "embedding_shape": list(embs.shape),
        "dtype": str(embs.dtype),
    }
    with open(out_info, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("\n[DONE]")
    print(f"- {out_emb}  shape={embs.shape}")
    print(f"- {out_chunks}")
    print(f"- {out_meta}")
    print(f"- {out_info}")


if __name__ == "__main__":
    main()