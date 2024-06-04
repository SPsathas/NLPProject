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

def get_relevant_documents(config_path: str, query: str, top_k: int):
    """
    Get relevant documents for a given query.
    Works on the musique dataset.
    Query should belong to the dev.json set (or shouldn't it?)
    """
    loader = RetrieverDataset("musiqueqa", "wiki-musiqueqa-corpus",
                              config_path,
                              Split.DEV)
    queries, _, corpus = loader.qrels()
    # print("queries", len(queries), len(qrels), len(corpus))
    print("First question:", query[0])
    print("First question:", query[1])
    tasb_search = Contriever(CONFIG_INSTANCE)

    similarity_measure = CosScore()
    response = tasb_search.retrieve(corpus, queries[:1], top_k, similarity_measure) # automatically looks for the already encoded corpus
    print("indices", len(response))
    # metrics = RetrievalMetrics(k_values=[1, 3, 5])
    # print(metrics.evaluate_retrieval(qrels=qrels, results=response))

    return response

if __name__ == "__main__":
    response = get_relevant_documents("./project_work_group_12/config.ini", # . == NLPProject
                           "Who is the mother of the director of film Polish-Russian War (Film)?",
                           3)
    print("response type:", type(response))
    print("response:", response)
    with open('response.json', 'w', encoding='utf-8') as f:
        json.dump(response, f, ensure_ascii=False, indent=4)
