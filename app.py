from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

from rag import ask_question
from faiss_indexer import create_faiss_index
from model import generate_description_for_image, classifier

from fpdf import FPDF

import io
import shutil
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

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],  # 허용할 출처 추가
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 가해자 모델
class Perpetrator(BaseModel):
    accidentDate: str  # 사고 발생 날짜
    accidentLocation: str  # 사고 발생 위치
    legalPlan: str  # 법적 대응 계획
    insuranceStatus: str  # 보험 가입 상태
    policeReport: bool  # 경찰 신고 여부
    settlementStatus: str  # 합의 내용
    injuryDescription: str  # 피해자 부상 내용
    accidentDetails : str # 사고 세부 사항
    scooterInfo: str  # 전동 킥보드 정보
    violationDetails: str  # 위반 내용

# 피해자 모델
class Victim(BaseModel):
    accidentDate: str  # 사고 발생 날짜
    accidentLocation: str  # 사고 발생 위치
    legalPlan: str  # 법적 대응 계획
    insuranceStatus: str  # 보험 가입 상태
    policeReport: bool  # 경찰 신고 여부
    settlementStatus: str  # 합의 내용
    injuryDescription: str  # 피해자 부상 내용
    accidentDetails : str # 사고 세부 사항
    scooterInfo: str  # 전동 킥보드 정보
    vehicleInfo : str # 가해 차량 정보
    perpetratorContact : bool # 가해자 연락처 공유 여부
    hasWitness : bool # 목격자 여부

# victim enpoint
@app.post("/ask/victim")
async def ask_victim_endpoint(participant: Victim):
    question = generate_victim_question(participant)
    answer = ask_question(question)
    return {"answer": answer}

# victim enpoint
@app.post("/ask/perpetrator")
async def ask_perpetrator_endpoint(participant: Perpetrator):
    question = generate_perpetrator_question(participant)
    answer = ask_question(question)
    return {"answer": answer}

# 챗봇 엔드포인트
@app.post("/chat")
async def ask_question_endpoint(request: QuestionRequest):
    question = request.question
    answer = ask_question(question)
    return {"answer": answer}

# Perpetrator와 Victim에 따라 질문을 생성하는 함수
def generate_perpetrator_question(participant: Perpetrator) -> str:
    # 공통 필드 처리
    question = (
        f"사고가 {participant.accidentDate}에 {participant.accidentLocation}에서 발생했습니다. "
        f"법적 대응 계획은 {participant.legalPlan}입니다. "
        f"피해자 부상 내용은 {participant.injuryDescription}이고, "
        f"사고 세부 사항은 {participant.accidentDetails}입니다. "
        f"전동 킥보드 정보는 {participant.scooterInfo}입니다. "
        f"보험 상태는 {participant.insuranceStatus}이며, "
        f"경찰 신고는 {'완료됨' if participant.policeReport else '신고하지 않음'}입니다. "
        f"합의 상태는 {participant.settlementStatus}입니다. "
    )

    question += (
            f"위반 내용은 {participant.violationDetails}입니다."
        )

    question += " 이러한 경우에는 어떤 법적 처리를 해야 할까요?"

    return question

def generate_victim_question(participant: Victim) -> str:
    # 공통 필드 처리
    question = (
        f"사고가 {participant.accidentDate}에 {participant.accidentLocation}에서 발생했습니다. "
        f"법적 대응 계획은 {participant.legalPlan}입니다. "
        f"피해자 부상 내용은 {participant.injuryDescription}이고, "
        f"사고 세부 사항은 {participant.accidentDetails}입니다. "
        f"전동 킥보드 정보는 {participant.scooterInfo}입니다. "
        f"보험 상태는 {participant.insuranceStatus}이며, "
        f"경찰 신고는 {'완료됨' if participant.policeReport else '신고하지 않음'}입니다. "
        f"합의 상태는 {participant.settlementStatus}입니다. "
    )

    question += (
            f"가해 차량 정보는 {participant.vehicleInfo}이고, "
            f"가해자 연락처는 {'받음' if participant.perpetratorContact else '받지 않음'}이며, "
            f"목격자는 {'존재함' if participant.hasWitness else '존재하지 않음'}입니다."
        )
        

    question += " 이러한 경우에는 어떤 법적 처리를 해야 할까요?"

    return question

# # POST 요청을 통해 질문을 받아 처리
# @app.post("/ask")
# async def ask_question_endpoint(request: QuestionRequest):
#     question = request.question
#     answer = ask_question(question)
#     return {"answer": answer}

# @app.post("/generate_text")
# async def get_description(request: ImageRequest):
#     try:
#         # 요청 바디에서 Image_path를 사용하여 설명 생성
#         image_url = request.Image_path

#         # 이미지 경로가 존재하는지 확인
#         if not os.path.exists(image_url):
#             raise HTTPException(status_code=404, detail="Image not found")
        
#         description = generate_description_for_image(image_url, classifier)

#         # 추가적인 정보 처리
#         additional_info = request.additional_info

#         # 설명과 추가 정보 반환
#         return {
#             "description": description
#         }
#     except FileNotFoundError as e:
#         raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    

class PDF(FPDF):
    def header(self):
        self.set_font('MaruBuri', '', 12)
        self.cell(0, 10, '사건 보고서', ln=True, align='C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('MaruBuri', '', 12)
        self.cell(0, 10, title, ln=True)
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('MaruBuri', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_chapter(self, title, body):
        self.add_page()
        self.chapter_title(title)
        self.chapter_body(body)

@app.post("/generate_text")
async def get_description(file: UploadFile = File(...)):
    try:
        # FPDF에서 폰트 등록
        pdf = PDF()
        pdf.add_font('MaruBuri', '', './font/MaruBuri.ttf', uni=True)

        # PDF 생성
        pdf.add_page()

        # 파일 데이터를 바이너리로 읽어들임
        file_data = await file.read()

        # 이미지 설명 생성 (이미지 데이터를 기반으로, 예시는 단순 텍스트 사용)
        description = generate_description_for_image(file_data, classifier)

        # PDF 생성 (디스크 내 임시 파일로 저장)
        pdf.multi_cell(0, 10, description)
        temp_pdf_path = "./uploads/description_report.pdf"  # 임시 PDF 파일 경로
        pdf.output(temp_pdf_path)

        # 파일을 디스크에 저장한 후, 해당 파일을 반환
        return FileResponse(temp_pdf_path, media_type="application/pdf", filename="description_report.pdf")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))