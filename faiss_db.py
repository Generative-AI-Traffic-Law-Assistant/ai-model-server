import faiss
import numpy as np

def init_faiss_index(dimension):
    """ FAISS 인덱스를 초기화하는 함수 """
    return faiss.IndexFlatL2(dimension)

def add_vectors_to_faiss(index, vectors):
    """ 벡터들을 FAISS 인덱스에 추가하는 함수 """
    index.add(np.array(vectors))

def search_faiss(index, query_vector, k=3):
    """ FAISS 인덱스에서 벡터 검색 (가장 관련된 k개의 결과 반환) """
    distances, indices = index.search(np.array(query_vector), k)
    return distances, indices