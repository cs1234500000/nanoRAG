import numpy as np
from transformers import AutoModel
from numpy.linalg import norm
from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os
from typing import List


class HFembedding:

    def __init__(self, path:str=''):
        self.path = path
        self.embedding=HuggingFaceEmbeddings(model_name=path)

    def get_embedding(self,content:str=''):
        return self.embedding.embed_query(content)

    def compare(self, text1: str, text2: str):
        embed1=self.embedding.embed_query(text1) 
        embed2=self.embedding.embed_query(text2)
        return np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))

    def compare_v(cls, vector1: List[float], vector2: List[float]) -> float:
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude


class OpenAIembedding:
    
    def __init__(self, path:str=''):
        self.path = path
        self.embedding=OpenAIEmbeddings()
    
    def get_embedding(self,content:str=''):
        content = content.replace("\n", " ")
        return self.embedding.embed_query(content)
    
    def compare(self, text1: str, text2: str):
        embed1=self.embedding.embed_query(text1) 
        embed2=self.embedding.embed_query(text2)
        return np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))
    
    def compare_v(cls, vector1: List[float], vector2: List[float]) -> float:
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude

