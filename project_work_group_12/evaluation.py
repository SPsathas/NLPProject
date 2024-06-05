from working_file import get_relevant_documents
from evaluate import load

def exact_match(predictions: list[str], references: list[str]) -> list[float]:
    """

    """
    exact_match_metric = load("exact_match")
    results = exact_match_metric.compute(predictions=predictions,
                                         references=references,
                                         ignore_case=True,
                                         ignore_punctuation=True)

    return results