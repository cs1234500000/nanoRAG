from tqdm import tqdm
import numpy as np 
from components.embeddings import HFembedding
import os
import json
from typing import List




class VectorDB:
    
    def __init__(self,docs:List=[]) -> None:
        self.docs = docs
    
    #get the vector embddings
    def get_vector(self,EmbeddingModel)->List[List[float]]:
        self.vectors = []
        for doc in tqdm(self.docs):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors
    
    #save vectors and documents in json 
    def persist(self,path:str='database')->None:
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/documents.json", 'w', encoding='utf-8') as f:
            json.dump(self.docs, f, ensure_ascii=False)
        with open(f"{path}/vectors.json", 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f)

    #load vectors and documents 
    def load_vector(self,path:str='database')->None:
        with open(f"{path}/vectors.json", 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)
        with open(f"{path}/document.json", 'r', encoding='utf-8') as f:
            self.document = json.load(f)
    
    #get cosine similarity
    def get_similarity(self, vector1: List[float], vector2: List[float],embedding_model) -> float:
        return embedding_model.compare_v(vector1, vector2)
    
    #get the similarity between query and embeddings 
    def query(self, query: str, EmbeddingModel, k: int = 1) -> List[str]:
        query_vector = EmbeddingModel.get_embedding(query)
        result = np.array([self.get_similarity(query_vector, vector,EmbeddingModel)
                          for vector in self.vectors])
        return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()