import json

from constants import Split
from data.datastructures.hyperparameters.dpr import DenseHyperParams
from data.loaders.RetrieverDataset import RetrieverDataset
from data.loaders.WikiMultihopQADataLoader import WikiMultihopQADataLoader
from metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from metrics.SimilarityMatch import CosineSimilarity as CosScore
from retriever.Contriever import Contriever

if __name__ == "__main__":
    config_instance = DenseHyperParams(query_encoder_path="facebook/contriever",
                                       document_encoder_path="facebook/contriever",
                                       batch_size=32)
    # config = config_instance.get_all_params()
    corpus_path = "C:/Users/steff/Documents/CS4360/NLPProject/wiki_musique_corpus.json"

    loader = RetrieverDataset("wikimultihopqa", "wiki-musiqueqa-corpus", "C:/Users/steff/Documents/CS4360/NLPProject/evaluation/config.ini", Split.DEV)
    queries, qrels, corpus = loader.qrels()
    print("queries", len(queries), len(qrels), len(corpus))
    tasb_search = Contriever(config_instance)

    ## wikimultihop

    # with open("/raid_data-lv/venktesh/BCQA/wiki_musique_corpus.json") as f:
    #     corpus = json.load(f)

    similarity_measure = CosScore()
    response = tasb_search.retrieve(corpus, queries, 100, similarity_measure)
    print("indices", len(response))
    metrics = RetrievalMetrics(k_values=[1, 3, 5])
    print(metrics.evaluate_retrieval(qrels=qrels, results=response))
