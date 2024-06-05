import json
import torch

from retriever.Contriever import Contriever
import openai
import os
from data.loaders.RetrieverDataset import RetrieverDataset
from data.loaders.MusiqueQaDataLoader import MusiqueQADataLoader
from constants import Split
from metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from metrics.SimilarityMatch import CosineSimilarity as CosScore
from data.datastructures.hyperparameters.dpr import DenseHyperParams

CONFIG_INSTANCE = DenseHyperParams(query_encoder_path="facebook/contriever",
                                     document_encoder_path="facebook/contriever",
                                     batch_size=16)

def get_relevant_documents(config_path: str, top_k: int):
    """
    Get relevant documents for a given query.
    Works on the musique dataset.
    Query should belong to the dev.json set (or shouldn't it?)
    """
    loader = RetrieverDataset("musiqueqa", "wiki-musiqueqa-corpus",
                              config_path,
                              Split.DEV)
    queries, qrels, corpus = loader.qrels()
    print("First question:", queries[0].text())
    tasb_search = Contriever(CONFIG_INSTANCE)

    similarity_measure = CosScore()
    response = tasb_search.retrieve(corpus, queries, top_k, similarity_measure) # automatically looks for the already encoded corpus
    print("indices", len(response))
    # metrics = RetrievalMetrics(k_values=[1, 3, 5])
    # print(metrics.evaluate_retrieval(qrels=qrels, results=response))

    processed_response = process_response(response)
    # final_response = get_textual_documents(processed_response, corpus)
    return processed_response#, final_response

def process_response(response):
    processed_response = {}
    for _, k in enumerate(response.keys()):
        docs = response[k]
        processed_response[k] = []
        for _, doc_id in enumerate(docs.keys()):
            processed_response[k].append(doc_id)

    return processed_response

def get_textual_documents(processed_response, corpus):
  # no more in use since modification in HfRetriever
    all_docs = []
    for _, query_id in enumerate(processed_response.keys()):
        all_docs.append(processed_response[query_id]) # so to iterate once in the corpus
    all_docs = [d for l in all_docs for d in l] # flatten list
    doc_text = {}

    print(corpus)
    for doc in corpus:
        if doc._idx in all_docs:
            doc_text[doc._idx] = doc._text

    for _, query_id in enumerate(processed_response.keys()):
        texts = []
        for doc_id in processed_response[query_id]:
          if doc_text[doc_id] not in texts: # trying to avoid duplicates
            texts.append(doc_text[doc_id])
        processed_response[query_id] = texts

    return processed_response

if __name__ == "__main__":
    top_k = 3
    response = get_relevant_documents("./project_work_group_12/config.ini", top_k)
    print(response)

