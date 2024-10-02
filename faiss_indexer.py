import faiss
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

# 여러 PDF 문서에서 데이터 임베딩 및 FAISS 인덱스 생성
def create_faiss_index():
    pdf_folder = './pdf/'
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    
    # 여러 PDF 문서 로드
    documents = []
    for pdf_file in pdf_files:
        loader = PyMuPDFLoader(os.path.join(pdf_folder, pdf_file))
        documents.extend(loader.load())

    # 텍스트를 조각내어 벡터화
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # OpenAI 임베딩 생성
    embeddings = OpenAIEmbeddings()

    # FAISS 벡터 저장소 생성
    faiss_index = FAISS.from_documents(docs, embeddings)

    # 인덱스 저장
    faiss_index.save_local("faiss_index")

# FAISS 인덱스 로드
def load_faiss_index():
    return FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

