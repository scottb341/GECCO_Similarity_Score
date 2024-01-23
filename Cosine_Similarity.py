# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:03:03 2024

@author: Scott Blyth
"""

import torch
from transformers import BertTokenizer, BertModel
from functools import lru_cache

CODE_BERT_MODEL = "microsoft/codebert-base"

class SimilarityScore: 
    def __init__(self, model_name):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name,output_hidden_states=True )
        self.model.eval()

    def cosine(self, v1,v2): 
        mag1 = torch.norm(v1)
        mag2 = torch.norm(v2)
        return torch.dot(v1, v2)/(mag1*mag2)
    
    def get_tokens(self, text):
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        return tokenized_text
    
    @lru_cache(maxsize=32, typed=False)
    def embed(self, text):
        tokenizer = self.tokenizer
        tokens = self.get_tokens(text)
        
        token_indices = tokenizer.convert_tokens_to_ids(tokens)
        
        segments_ids = [1] * len(tokens)
    
        tokens_tensor = torch.tensor([token_indices])
        segments_tensors = torch.tensor([segments_ids])
    
        with torch.no_grad():
            hidden_states,_ = self.model(tokens_tensor, segments_tensors)
            
        token_vecs = hidden_states[-2][0]
    
        # Calculate the average of all 22 token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
        
        return sentence_embedding
        
    def similarity(self, text1,text2):
        vec1 = self.embed(text1)
        vec2 = self.embed(text2)
        return self.cosine(vec1,vec2)
