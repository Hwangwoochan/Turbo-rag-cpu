import os
import pickle
import random
from llama_cpp import Llama
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# --- 1. 설정 ---
MODEL_PATH = "models/smollm_q4_k_m.gguf"
DOCS_DIR = "documents"
CACHE_DIR = "models/caches_3_random"
STORAGE_DIR = "doc_emb_random10"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROUP_SIZE = 3
SAMPLE_COUNT = 10  # 랜덤으로 뽑을 문서 개수

for path in [CACHE_DIR, STORAGE_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)

# --- 2. 초기화 ---
print(f"[STEP 1] 모델 로딩: {MODEL_PATH}")
llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_threads=4, verbose=False)
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

def build_random_index():
    # 전체 파일 목록 확보 및 랜덤 샘플링
    all_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".txt")]
    if len(all_files) < SAMPLE_COUNT:
        print(f"[알림] 문서가 {SAMPLE_COUNT}개보다 적어 전체 문서({len(all_files)}개)를 사용합니다.")
        sampled_files = all_files
    else:
        sampled_files = random.sample(all_files, SAMPLE_COUNT)
    
    sampled_files.sort() # 순서 정렬
    all_documents = []
    
    print(f"[STEP 2] 랜덤 추출된 {len(sampled_files)}개 문서를 {GROUP_SIZE}개씩 그룹화 시작...")

    # 샘플링된 파일을 3개씩 묶음 처리
    for i in range(0, len(sampled_files), GROUP_SIZE):
        batch_files = sampled_files[i : i + GROUP_SIZE]
        combined_content = ""
        group_names = ", ".join(batch_files)
        group_id = f"random_group_{i//GROUP_SIZE}"
        
        for fname in batch_files:
            with open(os.path.join(DOCS_DIR, fname), "r", encoding="utf-8") as f:
                combined_content += f"\n[Document: {fname}]\n" + f.read() + "\n"

        # (A) KV Cache 생성
        llm.reset()
        tokens = llm.tokenize(combined_content.encode('utf-8'))
        llm.eval(tokens)
        
        cache_path = os.path.abspath(os.path.join(CACHE_DIR, f"{group_id}.bin"))
        with open(cache_path, "wb") as f:
            pickle.dump(llm.save_state(), f)
        
        # (B) LlamaIndex용 문서 객체 생성
        doc = Document(
            text=combined_content,
            metadata={
                "group_id": group_id,
                "included_files": group_names,
                "kvcache_file_path": cache_path
            }
        )
        all_documents.append(doc)
        print(f" -> 생성 완료: [{group_names}] -> {group_id}.bin")

    # --- 3. 인덱스 생성 및 저장 ---
    print(f"[STEP 3] 벡터 인덱스 생성 중...")
    index = VectorStoreIndex.from_documents(all_documents)
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    
    print(f"\n[완료] {len(all_documents)}개의 그룹 인덱스가 {STORAGE_DIR}에 저장되었습니다.")

if __name__ == "__main__":
    build_random_index()