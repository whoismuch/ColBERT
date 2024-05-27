import torch
from transformers import BertModel, BertTokenizer
import numpy as np
import faiss
import os

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
    all_embeddings = []
    for doc_text in documents:
        embeddings = colbert_model.encode([doc_text]).squeeze(0)
        all_embeddings.append(embeddings.mean(dim=0).numpy())  # Средний вектор документа

    d = all_embeddings[0].shape[0]  # Размерность эмбеддингов

    quantizer = faiss.IndexFlatL2(d)  # Используется для кластеризации
    index = faiss.IndexIVFFlat(quantizer, d, 100)  # nlist - количество кластеров

    index.train(np.array(all_embeddings))  # Обучение кластеризатора
    index.add(np.array(all_embeddings))

    return index, np.array(all_embeddings)


def search_with_faiss(query, colbert_model, faiss_index, top_k=2):
    query_embedding = colbert_model.encode([query]).squeeze(0).mean(dim=0).numpy().reshape(1, -1)
    distances, indices = faiss_index.search(query_embedding, top_k)
    return indices[0], distances[0]


def get_top_k_documents(query, documents, colbert_model, faiss_index, top_k=2):
    top_k_indices, top_k_distances = search_with_faiss(query, colbert_model, faiss_index, top_k)
    top_k_docs = [(documents[idx], dist) for idx, dist in zip(top_k_indices, top_k_distances)]
    return top_k_docs


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    documents = [
        'The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated.',
        'The Manhattan Project and its atomic bomb helped bring an end to World War II. Its legacy of peaceful uses of atomic energy continues to have an impact on history and science.',
        'Essay on The Manhattan Project - The Manhattan Project The Manhattan Project was to see if making an atomic bomb possible. The success of this project would forever change the world forever making it known that something this powerful can be manmade.',
        'The Manhattan Project was the name for a project conducted during World War II, to develop the first atomic bomb. It refers specifically to the period of the project from 194 â\x80¦ 2-1946 under the control of the U.S. Army Corps of Engineers, under the administration of General Leslie R. Groves.']
    collection='./collection5mb.tsv'
    colbert_model = ColBERT(model_name='./model')
    faiss_index, document_embeddings = index_documents_with_faiss(documents, colbert_model)
    query = "test document"
    top_k_docs = get_top_k_documents(query, documents, colbert_model, faiss_index, top_k=2)
    for doc, distance in top_k_docs:
        print(f"Document: {doc} \nDistance: {distance}\n")
