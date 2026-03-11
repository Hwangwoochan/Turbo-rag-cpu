import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 환경 설정
model_path = "./final_merged_smollm2_rag"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 모델 로드 (Qwen2 대신 AutoModel 사용)
print(f"Loading model on {device}...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True
).to(device)
model.eval()

# 3. 토크나이저 로드 (에러 방지를 위해 use_fast=False 추가)
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
except Exception as e:
    print(f"Fast tokenizer failed, trying default: {e}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4. 추론 함수
def answer_question(context_chunks, query):
    system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    context_str = "".join(context_chunks)
    prompt = f"{system_prompt}<|im_start|>user\nDocs: {context_str}\n\nQuestion: {query}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # 입력 토큰 제외하고 답변만 추출
    response = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

# --- 실행 테스트 ---
if __name__ == "__main__":
    context = ["SmolLM2는 작지만 강력한 언어 모델입니다."]
    query = "SmolLM2가 뭐야?"
    
    print("Generating...")
    print(f"Result: {answer_question(context, query)}")