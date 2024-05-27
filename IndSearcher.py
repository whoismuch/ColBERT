import torch
from transformers import BertModel, BertTokenizer
import numpy as np
import faiss


class ColBERT:
    def __init__(self, model_name='bert-base-uncased'):
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def encode(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state
        return embeddings


def index_documents_with_faiss(documents, colbert_model):
    # Получение эмбеддингов для всех документов
    all_embeddings = []
    for doc_text in documents:
        embeddings = colbert_model.encode([doc_text]).squeeze(0)
        all_embeddings.append(embeddings.mean(dim=0).numpy())  # Средний вектор документа

    # Создание FAISS индекса
    d = all_embeddings[0].shape[0]  # Размерность эмбеддингов
    index = faiss.IndexFlatL2(d)
    index.add(np.array(all_embeddings))

    return index, np.array(all_embeddings)


def search_with_faiss(query, colbert_model, faiss_index, document_embeddings, top_k=2):
    query_embedding = colbert_model.encode([query]).squeeze(0).mean(dim=0).numpy().reshape(1, -1)

    # Поиск топ-k наиболее похожих документов
    distances, indices = faiss_index.search(query_embedding, top_k)

    return indices[0], distances[0]


# Пример использования:
documents = ["This is a test document.", "Another document for testing."]
query = "test document"

colbert_model = ColBERT()
faiss_index, document_embeddings = index_documents_with_faiss(documents, colbert_model)
top_k_indices, top_k_distances = search_with_faiss(query, colbert_model, faiss_index, document_embeddings, top_k=2)

print("Top-k document indices:", top_k_indices)
print("Top-k document distances:", top_k_distances)
