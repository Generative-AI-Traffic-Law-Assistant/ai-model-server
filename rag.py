from transformers import GPT2Tokenizer, GPT2LMHeadModel

# GPT2 모델과 토크나이저 로드
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_response(query, relevant_docs):
    """
    RAG 방식으로 질문과 검색된 문서를 기반으로 GPT 모델을 사용해 응답을 생성.
    
    :param query: 사용자가 입력한 질문
    :param relevant_docs: FAISS에서 검색된 관련 문서 목록
    :return: GPT 모델이 생성한 응답
    """

    # 검색된 문서를 하나로 결합하여 GPT 입력 생성
    combined_context = " ".join(relevant_docs)[:1024]  # GPT2 모델 입력 제한에 맞게 자르기
    
    # GPT2에 검색된 문서와 질문을 입력으로 넣어 답변 생성
    input_text = f"질문: {query}\n문서 정보: {combined_context}\n답변:"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, pad_token_id=tokenizer.eos_token_id)

    # GPT 모델의 답변 반환
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response