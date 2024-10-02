from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from faiss_indexer import load_faiss_index
from openai_api import get_openai_api_key

# GPT 모델 설정
llm = ChatOpenAI(api_key=get_openai_api_key(), model_name="gpt-3.5-turbo", temperature=0)

# 프롬프트 템플릿 정의
template = """
전동킥보드 법률 도우미 챗봇입니다. 전동킥보드 외의 질문을 받으면 정중하게 대답을 할 수 없다고 답하세요. 

질문에 대한 답변을 제공하기 전에, 관련 법률 문서들을 검색하여 그 내용과 함께 답변을 드리겠습니다.

질문: {question}

검색된 문서:
{context}

답변:
"""

prompt = PromptTemplate(template=template, input_variables=["question", "context"])

# 검색 기반 QA 체인 구성
def ask_question(question):
    faiss_index = load_faiss_index()
    docs = faiss_index.similarity_search(question)
    context = "\n".join([doc.page_content for doc in docs])

    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=faiss_index.as_retriever())
    
    return qa_chain.invoke(question)
