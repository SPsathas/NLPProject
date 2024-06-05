from working_file import get_relevant_documents
from evaluate import load
from gfw import DevDataset, GPTQA

def exact_match(predictions: list[str], references: list[str]) -> list[float]:
    """
    """
    exact_match_metric = load("exact_match")
    results = exact_match_metric.compute(predictions=predictions,
                                         references=references,
                                         ignore_case=True,
                                         ignore_punctuation=True)

    return sum(results)/len(results)

def evaluate(dev: DevDataset, mode: str = 'oracle', retrieved_contexts: dict =None):
    ground_truth = []
    answers = []
    gpt = GPTQA()

    for question in dev:
        if mode == 'oracle':
            context = question.get_contexts('gold')
        elif mode == 'retrieved' and retrieved_contexts is not None:
            context = retrieved_contexts.get(question.id)
        else: raise ValueError("Something wrong happened")

        response = gpt.ask_question(question, context=context)
        ground_truth.append(question.answer)
        answers.append(response[1])

    return exact_match(answers, ground_truth)

if __name__ == "__main__":
    dev = DevDataset("./project_work_group_12/data/dev.json")
    top_k = 3
    retrieved_documents = get_relevant_documents("./project_work_group_12/config.ini", top_k)

    print("Result with oracle contexts:", evaluate(dev, 'oracle'))
    print("Result with retrieved contexts:", evaluate(dev, 'retrieved'), retrieved_documents)