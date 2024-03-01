# GECCO_Similarity_Score

## Description 
Given two code-snippets (texts), it computes the semantic similarity between the 2 texts using the cosine of the vector embeddings.

## Installation 

```pip install transformers```
```pip install torch```

## Example Use Case
```python
s = SimilarityScore(CODE_BERT_MODEL) # CODE_BERT_MODEL = "microsoft/codebert-base"
v1 = s.embed("x=10")
v2 = s.embed("y=10")
print(s.similarity(v1,v2))
```
This should print ```tensor(0.9967)```
