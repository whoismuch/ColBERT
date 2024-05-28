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


class IndexerRetriever:
    def __init__(self, documents, model_name='bert-base-uncased'):
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.documents = documents

    def encode(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state
        return embeddings


    def index_documents_with_faiss(self, nlist=2, sample_ratio=0.3, m=8):
        all_embeddings = []
        doc_embeds = []
        for doc_text in self.documents:
            embeddings = self.encode([doc_text]).squeeze(0)
            doc_embeds.append(embeddings)
            all_embeddings.append(embeddings.mean(dim=0).numpy())  # Средний вектор документа

        # Случайная выборка подмножества данных для обучения
        num_samples = max(1, int(len(all_embeddings) * sample_ratio))
        sampled_embeddings = random.sample(all_embeddings, num_samples)

        d = all_embeddings[0].shape[0]  # Размерность эмбеддингов

        quantizer = faiss.IndexFlatL2(d)  # Используется для кластеризации
        # index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)  # nlist - количество кластеров, m - количество подпространств
        index = faiss.IndexIVFFlat(quantizer, d, nlist)  # nlist - количество кластеров, m - количество подпространств

        index.train(np.array(sampled_embeddings))  # Обучение кластеризатора на подмножестве данных
        index.add(np.array(all_embeddings))  # Добавление всех данных в индекс

        self.faiss_index = index
        self.doc_embeddings = doc_embeds
        return index, np.array(all_embeddings)


    def search_with_faiss(self, query, top_k=2):
        query_embedding = self.encode([query]).squeeze(0).mean(dim=0).numpy().reshape(1, -1)
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        return indices[0], distances[0]


    def get_top_k_documents(self, query, top_k=2):
        top_k_indices, top_k_distances = self.search_with_faiss(query, top_k)
        top_k_docs = [(self.documents[idx], dist) for idx, dist in zip(top_k_indices, top_k_distances)]
        top_k_embeds_indices = [(self.doc_embeddings[idx], idx) for idx in top_k_indices]
        return top_k_docs, top_k_embeds_indices

    def score_with_colbert(self, query, doc_embed):
        query_embeddings = self.encode([query]).squeeze(0)

        # Рассчитываем косинусное сходство между каждым токеном в запросе и каждом токеном в документе
        similarity_matrix = cosine_similarity(query_embeddings.unsqueeze(1), torch.tensor(doc_embed).float().unsqueeze(0))

        # Суммируем максимальные значения для каждого токена в запросе
        score = similarity_matrix.max(dim=1).values.sum().item()

        return score


    def rerank_documents_with_colbert(self, query, doc_embeds_indices):
        # Скоринг документов
        scored_documents = [(self.documents[idx], self.score_with_colbert(query, doc_embed)) for doc_embed, idx in doc_embeds_indices]

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

    documents = load_documents_from_tsv('./collection50lines.tsv')

    indRetriever = IndexerRetriever(documents, model_name='./model')
    faiss_index, document_embeddings = indRetriever.index_documents_with_faiss()

    query = "what is The Manhattan Project"
    _, top_k_embeds_indices = indRetriever.get_top_k_documents(query, top_k=10)

    reranked_docs = indRetriever.rerank_documents_with_colbert(query, top_k_embeds_indices)

    for doc, distance in reranked_docs:
        print(f"Document: {doc} \nDistance: {distance}\n")

    # documents = load_documents_from_tsv('collection5mb.tsv')
    # colbert_model = ColBERT(model_name=namespace.checkpoint_path)
    # faiss_index, document_embeddings = index_documents_with_faiss(documents, colbert_model)
    #
    # query = "test document"
    # preselected_docs = get_top_k_documents(query, documents, colbert_model, faiss_index, top_k=5)
    #
    # # Извлечение текстов документов для точного скоринга
    # preselected_texts = [doc for doc, _ in preselected_docs]
    #
    # reranked_docs = rerank_documents_with_colbert(query, preselected_texts, colbert_model)
    #
    # for doc, distance in reranked_docs:
    #     print(f"Document: {doc} \nDistance: {distance}\n")
