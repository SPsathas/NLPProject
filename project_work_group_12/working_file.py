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
import random

CONFIG_INSTANCE = DenseHyperParams(query_encoder_path="facebook/contriever",
                                     document_encoder_path="facebook/contriever",
                                     batch_size=16)

def get_random_pairs(response, num_pairs):
    # Extract the dictionary from the response
    inner_dict = next(iter(response.values()))

    # Get a list of keys from the dictionary
    keys = list(inner_dict.keys())

    # Select random keys
    random_keys = random.sample(keys, num_pairs)

    # Create a new dictionary with the selected key-value pairs
    random_pairs = {key: inner_dict[key] for key in random_keys}

    return random_pairs

def get_relevant_documents(config_path: str, query: str, top_k: int, inverted = False):
    """
    Get relevant documents for a given query.
    Works on the musique dataset.
    Query should belong to the dev.json set (or shouldn't it?)
    """
    loader = RetrieverDataset("musiqueqa", "wiki-musiqueqa-corpus",
                              config_path,
                              Split.DEV)
    queries, qrels, corpus = loader.qrels()
    # print("queries", len(queries), len(qrels), len(corpus))
    print("First question:", queries[0].text())
    print("First question:", queries[1].text())
    tasb_search = Contriever(CONFIG_INSTANCE)

    similarity_measure = CosScore(inverted)
    response = tasb_search.retrieve(corpus, queries[:1], top_k, similarity_measure) # automatically looks for the already encoded corpus
    print("indices", len(response))
    metrics = RetrievalMetrics(k_values=[1, 3, 5])
    print(metrics.evaluate_retrieval(qrels=qrels, results=response))

    return response

def f(response):
    processed_response = {}
    for _, k in enumerate(response.keys()):
        docs = response[k]
        processed_response[k] = []
        for _, doc_id in enumerate(docs.keys()):
            processed_response[k].append(doc_id)

    return processed_response

def g(processed_response, corpus):
    all_docs = []
    for _, query_id in enumerate(processed_response.keys()):
        all_docs.append(processed_response[query_id]) # so to iterate once in the corpus
    all_docs = [d for l in all_docs for d in l] # flatten list
    doc_text = {}

    for _, doc_id in enumerate(corpus):
        if doc_id in all_docs:
            doc_text[doc_id] = corpus[doc_id]['text']

    for _, query_id in enumerate(processed_response.keys()):
        texts = []
        for _, doc_id in enumerate(processed_response[query_id]):
            texts.append(doc_text[doc_id])
        processed_response[query_id] = texts

    return processed_response

if __name__ == "__main__":
    response = get_relevant_documents("./project_work_group_12/config.ini", # . == NLPProject
                           "Who is the mother of the director of film Polish-Russian War (Film)?",
                           3)
    response_inverted = get_relevant_documents("./project_work_group_12/config.ini", # . == NLPProject
                           "Who is the mother of the director of film Polish-Russian War (Film)?",
                           3, inverted = True)

    random_pairs = get_random_pairs(response_inverted, 2)

    response.update(random_pairs)
    print(response)
    with open('response.json', 'w', encoding='utf-8') as f:
        json.dump(response, f, ensure_ascii=False, indent=4)
