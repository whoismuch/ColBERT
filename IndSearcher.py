import torch
from transformers import BertModel, BertTokenizer
import numpy as np
import faiss
import os
from torch.nn.functional import cosine_similarity
import pandas as pd
import random
import sys
import argparse


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


def index_documents_with_faiss(documents, colbert_model, nlist=100, sample_ratio=0.1, m=8):
    all_embeddings = []
    for doc_text in documents:
        embeddings = colbert_model.encode([doc_text]).squeeze(0)
        all_embeddings.append(embeddings.mean(dim=0).numpy())  # Средний вектор документа

    # Случайная выборка подмножества данных для обучения
    num_samples = max(1, int(len(all_embeddings) * sample_ratio))
    sampled_embeddings = random.sample(all_embeddings, num_samples)

    d = all_embeddings[0].shape[0]  # Размерность эмбеддингов

    quantizer = faiss.IndexFlatL2(d)  # Используется для кластеризации
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)  # nlist - количество кластеров, m - количество подпространств

    index.train(np.array(sampled_embeddings))  # Обучение кластеризатора на подмножестве данных
    index.add(np.array(all_embeddings))  # Добавление всех данных в индекс

    return index, np.array(all_embeddings)


def search_with_faiss(query, colbert_model, faiss_index, top_k=2):
    query_embedding = colbert_model.encode([query]).squeeze(0).mean(dim=0).numpy().reshape(1, -1)
    distances, indices = faiss_index.search(query_embedding, top_k)
    return indices[0], distances[0]


def get_top_k_documents(query, documents, colbert_model, faiss_index, top_k=2):
    top_k_indices, top_k_distances = search_with_faiss(query, colbert_model, faiss_index, top_k)
    top_k_docs = [(documents[idx], dist) for idx, dist in zip(top_k_indices, top_k_distances)]
    return top_k_docs


def score_with_colbert(query, document, colbert_model):
    # Кодируем запрос и документ
    query_embeddings = colbert_model.encode([query]).squeeze(0)
    doc_embeddings = colbert_model.encode([document]).squeeze(0)

    # Рассчитываем косинусное сходство между каждым токеном в запросе и каждом токеном в документе
    similarity_matrix = cosine_similarity(query_embeddings.unsqueeze(1), doc_embeddings.unsqueeze(0))

    # Суммируем максимальные значения для каждого токена в запросе
    scores = similarity_matrix.max(dim=1).values.sum().item()

    return scores


def rerank_documents_with_colbert(query, documents, colbert_model):
    # Скоринг документов
    scored_documents = [(doc, score_with_colbert(query, doc, colbert_model)) for doc in documents]
    # Сортировка документов по убыванию скорингов
    scored_documents = sorted(scored_documents, key=lambda x: x[1], reverse=True)

    return scored_documents


def load_documents_from_tsv(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None)
    documents = df[1].tolist()  # Второй столбец содержит тексты документов
    return documents

def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument ('--collection_path', default='./collection5mb.tsv')
    parser.add_argument ('--checkpoint_path', default='./model')
    return parser

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    collection = namespace.collection_path
    documents = load_documents_from_tsv(collection)
    colbert_model = ColBERT(model_name='./model')
    faiss_index, document_embeddings = index_documents_with_faiss(documents, colbert_model)
    query = "test document"
    preselected_docs = get_top_k_documents(query, documents, colbert_model, faiss_index, top_k=10)

    # Извлечение текстов документов для точного скоринга
    preselected_texts = [doc for doc, _ in preselected_docs]

    reranked_docs = rerank_documents_with_colbert(query, preselected_texts, colbert_model)

    for doc, distance in reranked_docs:
        print(f"Document: {doc} \nDistance: {distance}\n")
