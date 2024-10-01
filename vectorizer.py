from sentence_transformers import SentenceTransformer

# all-MiniLM-L6-v2 모델 사용
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def text_to_vector(texts):
    return embedder.encode(texts)