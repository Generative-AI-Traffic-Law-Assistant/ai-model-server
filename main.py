from fastapi import FastAPI
from pdf_extractor import extract_text_from_pdf
from vectorizer import text_to_vector
from faiss_db import init_faiss_index, add_vectors_to_faiss, search_faiss
from rag import generate_response

import os

app = FastAPI()

# 전역 변수로 FAISS 인덱스 및 PDF 텍스트 보관
index = None
pdf_texts = []

@app.on_event("startup")
async def load_data():
    """ 서버가 시작할 때 PDF 파일을 불러와 벡터화하고 FAISS 인덱스에 저장 """
    global index, pdf_texts

    pdf_dir = "/content/drive/MyDrive/pdf_folder/"
    
    # PDF 파일에서 텍스트 추출
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            text = extract_text_from_pdf(pdf_path)
            pdf_texts.append(text)
    
    # 텍스트 벡터화
    pdf_vectors = text_to_vector(pdf_texts)
    
    # FAISS 인덱스 초기화 및 벡터 추가
    dimension = pdf_vectors.shape[1]
    index = init_faiss_index(dimension)
    add_vectors_to_faiss(index, pdf_vectors)

@app.post("/chatbot/")
async def legal_chatbot(query: str):
    """ 사용자의 질문에 따라 FAISS 인덱스에서 관련 법률 정보를 검색하여 응답 """
    query_vector = text_to_vector([query])
    distances, indices = search_faiss(index, query_vector)
    
    # 검색된 문서 출력 (최대 3개)
    relevant_docs = [pdf_texts[idx] for idx in indices[0]]
    
    response = generate_response(query, relevant_docs)
    
    return {"response": response}
