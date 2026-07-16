import faiss
import numpy as np
import uuid
import pickle
import os
from utils.logger import logger

class VectorStore:
    def __init__(self, index_file="faiss_index.bin", mapping_file="uuid_mapping.pkl", dim=512):
        self.index_file = index_file
        self.mapping_file = mapping_file
        self.dim = dim
        self.uuid_mapping = []  # Maps FAISS index to UUID
        self.index = None
        self._initialize_index()

    def _initialize_index(self):
        # We use Inner Product for Cosine Similarity (assuming normalized vectors)
        if os.path.exists(self.index_file):
            logger.info("Loading existing FAISS index...")
            self.index = faiss.read_index(self.index_file)
            if os.path.exists(self.mapping_file):
                with open(self.mapping_file, "rb") as f:
                    self.uuid_mapping = pickle.load(f)
        else:
            logger.info("Creating new FAISS index...")
            self.index = faiss.IndexFlatIP(self.dim)
            self.uuid_mapping = []

    def save(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.mapping_file, "wb") as f:
            pickle.dump(self.uuid_mapping, f)

    def add_embedding(self, embedding: np.ndarray, visitor_uuid: uuid.UUID):
        embedding = np.array([embedding], dtype=np.float32)
        self.index.add(embedding)
        self.uuid_mapping.append(visitor_uuid)
        self.save()

    def search(self, embedding: np.ndarray, threshold: float = 0.7):
        if self.index.ntotal == 0:
            return None, 0.0

        embedding = np.array([embedding], dtype=np.float32)
        distances, indices = self.index.search(embedding, 1)
        
        best_score = distances[0][0]
        best_idx = indices[0][0]

        if best_score > threshold and best_idx != -1:
            return self.uuid_mapping[best_idx], best_score
        return None, best_score
