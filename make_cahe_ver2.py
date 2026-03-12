import os
import pickle
from llama_cpp import Llama
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

MODEL_PATH = "models/smollm_q4_k_m.gguf"
DOCS_DIR = "documents"
CACHE_DIR = "models/caches"
STORAGE_DIR = "doc_emb"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PREFIX_TEMPLATE = "Docs:\n{doc}\n\nQuestion:"

for path in [CACHE_DIR, STORAGE_DIR]:
    os.makedirs(path, exist_ok=True)

print(f"[1/4] 모델 로딩 중: {MODEL_PATH}")
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, verbose=False)

Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

def build_everything():
    all_documents = []
    print("[2/4] KV Cache 생성 및 메타데이터 구성 시작...")

    for filename in os.listdir(DOCS_DIR):
        if not filename.endswith(".txt"):
            continue

        doc_path = os.path.join(DOCS_DIR, filename)
        cache_path = os.path.abspath(os.path.join(CACHE_DIR, f"{filename}.bin"))

        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 캐시 생성용 prefix를 명시적으로 고정
        prefix_text = PREFIX_TEMPLATE.format(doc=content)

        llm.reset()
        prefix_tokens = llm.tokenize(prefix_text.encode("utf-8"))
        llm.eval(prefix_tokens)

        state = llm.save_state()
        with open(cache_path, "wb") as f:
            pickle.dump(state, f)

        doc = Document(
            text=content,
            metadata={
                "file_name": filename,
                "kvcache_file_path": cache_path,
                "prefix_template": PREFIX_TEMPLATE,
            }
        )
        all_documents.append(doc)
        print(f" -> 완료: {filename}")

    print(f"[3/4] '{EMBED_MODEL}' 모델로 임베딩 및 인덱싱 중...")
    index = VectorStoreIndex.from_documents(all_documents)

    print(f"[4/4] 인덱스 저장: {STORAGE_DIR}")
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    print("\n[성공] 인덱싱과 캐시 생성이 모두 완료되었습니다.")

if __name__ == "__main__":
    build_everything()