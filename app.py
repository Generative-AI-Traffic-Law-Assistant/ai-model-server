from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel

from rag import ask_question
from faiss_indexer import create_faiss_index


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

app = FastAPI(lifespan=lifespan)

# POST 요청을 통해 질문을 받아 처리
@app.post("/ask")
async def ask_question_endpoint(request: QuestionRequest):
    question = request.question
    answer = ask_question(question)
    return {"answer": answer}