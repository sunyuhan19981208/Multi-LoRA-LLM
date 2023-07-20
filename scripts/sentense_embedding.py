import pdb
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

embedding_model = SentenceTransformer('/home/sunyuhan/syh/sunyuhan/exp/Multi-LoRA-LLM/scripts/all-MiniLM-L6-v2')
embeddings = embedding_model.encode(sentences)
pdb.set_trace()
print(embeddings)
