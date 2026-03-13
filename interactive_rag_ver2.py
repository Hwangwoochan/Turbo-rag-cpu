import os
import time
import pickle
import argparse
from llama_cpp import Llama
from tabulate import tabulate

from llama_index.core import Settings, load_index_from_storage, StorageContext, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

PREFIX_TEMPLATE = "Docs:\n{doc}\n\nQuestion:"
NO_CACHE_FULL_TEMPLATE = "Docs:\n{doc}\n\nQuestion: {query}\nAnswer:"
CACHE_SUFFIX_TEMPLATE = " {query}\nAnswer:"


def build_retriever(storage_dir: str, embedding_model_name: str, top_k: int):
    print(f"[STEP] 리트리버 로딩 중... (모델: {embedding_model_name})")
    Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    index = load_index_from_storage(storage_context)
    return index.as_retriever(similarity_top_k=top_k)


def run_interactive_rag(args):
    print(f"[STEP] LLM 로딩 중: {args.model_path}")
    llm = Llama(
        model_path=args.model_path,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        verbose=False
    )

    retriever = build_retriever(args.storage_dir, args.embedding_model, args.top_k)

    print("\n[READY] Turbo-RAG 시스템이 준비되었습니다.")
    print("종료하려면 /exit 또는 /quit를 입력하세요.\n")

    while True:
        query = input("Q> ").strip()
        if not query:
            continue
        if query in ("/exit", "/quit"):
            break

        stats = {}

        # 전체 요청 시작 시각
        t_request_start = time.perf_counter()

        # STEP 1: Retrieval
        t0 = time.perf_counter()
        query_bundle = QueryBundle(query_str=query)
        retrieved_nodes = retriever.retrieve(query_bundle)
        stats["retrieval_time"] = time.perf_counter() - t0

        if not retrieved_nodes:
            print("관련 문서를 찾지 못했습니다.")
            continue

        best_node = retrieved_nodes[0].node
        cache_path = best_node.metadata.get("kvcache_file_path")

        # STEP 2: Cache Load
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

        stats["cache_load_time"] = time.perf_counter() - t1

        # STEP 3: Prompt Build
        if cache_loaded:
            prompt = CACHE_SUFFIX_TEMPLATE.format(query=query)
        else:
            prompt = NO_CACHE_FULL_TEMPLATE.format(doc=best_node.text, query=query)

        tokens = llm.tokenize(prompt.encode("utf-8"))

        # STEP 4: Generation
        t_gen_start = time.perf_counter()
        first_token_timestamp = None
        last_token_timestamp = None

        generated_text = ""
        token_count = 0

        for token in llm.generate(tokens, temp=args.temp):
            now = time.perf_counter()

            if token_count == 0:
                first_token_timestamp = now

            token_str = llm.detokenize([token]).decode("utf-8", errors="ignore")
            generated_text += token_str
            token_count += 1
            last_token_timestamp = now

            if token_count >= args.max_tokens or token == llm.token_eos():
                break

        t_request_end = time.perf_counter()

        # ----- Metrics 계산 -----
        # 1) 전체 파이프라인 기준 TTFT
        if first_token_timestamp is not None:
            stats["ttft"] = first_token_timestamp - t_request_start
        else:
            stats["ttft"] = 0.0

        # 2) generate() 호출부터 마지막 토큰까지
        stats["total_gen_time"] = t_request_end - t_gen_start

        # 3) 첫 토큰 이후 마지막 토큰까지 시간
        if first_token_timestamp is not None and last_token_timestamp is not None and token_count >= 2:
            stats["first_to_last"] = last_token_timestamp - first_token_timestamp
        else:
            stats["first_to_last"] = 0.0

        # 4) TPOT: 첫 토큰 이후 평균 토큰당 시간
        if token_count > 1:
            stats["tpot"] = stats["first_to_last"] / (token_count - 1)
        else:
            stats["tpot"] = 0.0

        # 5) 전체 E2E
        stats["total_e2e"] = t_request_end - t_request_start

        doc_token_count = len(llm.tokenize(best_node.text.encode("utf-8")))

        print(f"\n[ANSWER]\n{generated_text.strip()}")
        print("\n" + "=" * 50)
        print(f"[*] 검색된 문서 파일: {best_node.metadata.get('file_name', 'N/A')}")
        print(f"[*] 검색된 문서 토큰 수: {doc_token_count}")
        print(f"[*] 생성된 토큰 수: {token_count}")
        print(f"[*] 캐시 적용 여부: {'YES' if cache_loaded else 'NO'}")

        perf_data = [
            ["Retrieval", f"{stats['retrieval_time']:.4f}s"],
            ["Cache Load", f"{stats['cache_load_time']:.4f}s"],
            ["TTFT", f"{stats['ttft']:.4f}s"],
            ["TPOT", f"{stats['first_to_last']:.4f}s"],
            ["Gen per tokens", f"{stats['tpot']:.4f}s/token"],
            ["Gen Total (generate 전체)", f"{stats['total_gen_time']:.4f}s"],
            ["Total E2E", f"{stats['total_e2e']:.4f}s"],
        ]
        print(tabulate(perf_data, headers=["Phase", "Time"], tablefmt="simple"))
        print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--storage_dir", type=str, default="doc_emb")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--n_ctx", type=int, default=2048)
    parser.add_argument("--n_threads", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temp", type=float, default=0.1)

    args = parser.parse_args()
    run_interactive_rag(args)