import numpy as np

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    L2 Normalizes a vector embedding for Cosine Similarity search in FAISS.
    FAISS Inner Product (IP) on L2 normalized vectors is equivalent to Cosine Similarity.
    """
    embedding = np.array(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding
