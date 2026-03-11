import os
import time
import pickle
import argparse
from typing import List, Tuple
from llama_cpp import Llama
from tabulate import tabulate

# LlamaIndex (Retrieval용)
from llama_index.core import Settings, load_index_from_storage, StorageContext, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def build_retriever(storage_dir: str, embedding_model_name: str, top_k: int):
    print(f"[STEP] 리트리버 로딩 중... (모델: {embedding_model_name})")
    Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    index = load_index_from_storage(storage_context)
    return index.as_retriever(similarity_top_k=top_k)

def run_interactive_rag(args):
    # 1. 모델 초기화 (CPU 최적화)
    print(f"[STEP] LLM 로딩 중: {args.model_path}")
    llm = Llama(
        model_path=args.model_path,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        verbose=False
    )

    # 2. 리트리버 초기화
    retriever = build_retriever(args.storage_dir, args.embedding_model, args.top_k)

    print("\n[READY] Turbo-RAG 시스템이 준비되었습니다.")
    print("종료하려면 /exit 또는 /quit를 입력하세요.\n")

    while True:
        query = input("Q> ").strip()
        if not query: continue
        if query in ("/exit", "/quit"): break

        stats = {}
        
        # --- [STEP 1] Retrieval ---
        t0 = time.perf_counter()
        query_bundle = QueryBundle(query_str=query)
        retrieved_nodes = retriever.retrieve(query_bundle)
        stats['retrieval_time'] = time.perf_counter() - t0
        
        if not retrieved_nodes:
            print("관련 문서를 찾지 못했습니다.")
            continue

        best_node = retrieved_nodes[0].node # Top-1 사용
        cache_path = best_node.metadata.get("kvcache_file_path")

        # --- [STEP 2] KV Cache Loading ---
        t1 = time.perf_counter()
        llm.reset()
        cache_loaded = False
        if args.use_cache and cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    state = pickle.load(f)
                llm.load_state(state)
                cache_loaded = True
            except Exception as e:
                print(f"[WARN] 캐시 로드 실패: {e}")
        
        stats['cache_load_time'] = time.perf_counter() - t1

        # --- [STEP 3] Prompting & Generation ---
        # 캐시를 사용했다면 질문 부분만, 아니면 전체를 프롬프트로 구성
        if cache_loaded:
            prompt = f"\n\nQuestion: {query}\nAnswer:"
        else:
            prompt = f"Docs: {best_node.text}\n\nQuestion: {query}\nAnswer:"

        tokens = llm.tokenize(prompt.encode('utf-8'))
        
        t_gen_start = time.perf_counter()
        first_token_time = 0
        generated_text = ""
        token_count = 0

        # 토큰 생성 루프
        for token in llm.generate(tokens, temp=args.temp):
            if token_count == 0:
                first_token_time = time.perf_counter() - t_gen_start
            
            token_str = llm.detokenize([token]).decode('utf-8', errors='ignore')
            generated_text += token_str
            token_count += 1
            
            # 실시간 답변 출력 (선택 사항)
            # print(token_str, end="", flush=True)

            if token_count >= args.max_tokens or token == llm.token_eos():
                break
        
        stats['ttft'] = first_token_time
        stats['total_gen_time'] = time.perf_counter() - t_gen_start
        stats['tpot'] = (stats['total_gen_time'] - stats['ttft']) / (token_count - 1) if token_count > 1 else 0
        stats['total_e2e'] = time.perf_counter() - t0

        # --- [RESULT 출력] ---
        # --- [RESULT 출력] ---
        print(f"\n[ANSWER]\n{generated_text.strip()}")
        
        # 🔹 검색된 문서 토큰 수 계산
        doc_tokens = llm.tokenize(best_node.text.encode("utf-8"))
        doc_token_count = len(doc_tokens)

        print("\n" + "="*50)
        print(f"[*] 검색된 문서 토큰 수: {doc_token_count}")
        print(f"[*] 캐시 적용 여부: {'YES' if cache_loaded else 'NO'}")
        
        perf_data = [
            ["Retrieval", f"{stats['retrieval_time']:.4f}s"],
            ["Cache Load", f"{stats['cache_load_time']:.4f}s"],
            ["TTFT (첫 토큰)", f"{stats['ttft']:.4f}s"],
            ["TPOT (토큰당)", f"{stats['tpot']:.4f}s"],
            ["Total E2E", f"{stats['total_e2e']:.4f}s"]
        ]
        print(tabulate(perf_data, headers=["Phase", "Time"], tablefmt="simple"))
        print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--storage_dir", type=str, default="doc_emb")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--top_k", type=int, default=1) # 현재는 Top-1 캐시 로드 위주
    parser.add_argument("--use_cache", action="store_true", default=True)
    parser.add_argument("--n_ctx", type=int, default=2048)
    parser.add_argument("--n_threads", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temp", type=float, default=0.1)
    
    args = parser.parse_args()
    run_interactive_rag(args)