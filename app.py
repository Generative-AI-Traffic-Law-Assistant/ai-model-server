from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel

from rag import ask_question
from faiss_indexer import create_faiss_index
from model import generate_description_for_image, classifier

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 서버 시작 전에 FAISS 인덱스를 생성
@asynccontextmanager
async def lifespan(app: FastAPI):
    create_faiss_index()
    print("FAISS 인덱스 생성 완료!")

    yield
    print("서버 종료...")

    
# 질문을 포함한 JSON 요청을 위한 Pydantic 모델 정의
class QuestionRequest(BaseModel):
    question: str

class ImageRequest(BaseModel):
    Image_path: str
    additional_info: str

app = FastAPI(lifespan=lifespan)

# POST 요청을 통해 질문을 받아 처리
@app.post("/ask")
async def ask_question_endpoint(request: QuestionRequest):
    question = request.question
    answer = ask_question(question)
    return {"answer": answer}

@app.post("/generate_text")
async def get_description(request: ImageRequest):
    try:
        # 요청 바디에서 Image_path를 사용하여 설명 생성
        image_url = request.Image_path
        description = generate_description_for_image(image_url, classifier)

        # 추가적인 정보 처리
        additional_info = request.additional_info

        # 설명과 추가 정보 반환
        return {
            "description": description
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))