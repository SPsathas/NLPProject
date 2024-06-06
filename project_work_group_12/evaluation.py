from working_file import get_relevant_documents
from evaluate import load
from gfw import DevDataset, GPTQA
import json

def exact_match(predictions: list[str], references: list[str]) -> list[float]:
    """
    """
    exact_match_metric = load("exact_match")
    results = exact_match_metric.compute(predictions=predictions,
                                         references=references,
                                         ignore_case=True,
                                         ignore_punctuation=True)

    return results['exact_match']

def evaluate(dev: DevDataset, mode: str = 'oracle', retrieved_contexts: dict =None, write_to_file: tuple[bool, str] = (False, 'out.txt')):
    ground_truths = []
    answers = []
    prompts = []

    gpt = GPTQA()

    for question in dev:
        if mode == 'oracle':
            context = question.get_contexts('gold')
        elif mode == 'retrieved' and retrieved_contexts is not None:
            context = retrieved_contexts.get(question.id)
        else: raise ValueError("Something wrong happened")

        response = gpt.ask_question(question, context=context)
        ground_truths.append(question.answer)
        answers.append(response[1])
        prompts.append(response[0])

    if write_to_file[0]:
        with open(write_to_file[1], 'w') as f:
            for prompt, response, ground_truth in zip(prompts, answers, ground_truths):
                f.write(f"_____Prompt_____\n{prompt}\n\n_____Response_____\n{response}\n\n_____Ground Truth_____\n{ground_truth}\n\n\n")
                
    return exact_match(answers, ground_truths)

if __name__ == "__main__":
    dev = DevDataset("./project_work_group_12/data/dev.json")
    top_k = 3
    f = open(f"results_{top_k}.json")
    retrieved_documents = json.load(f)
    print(retrieved_documents)

    print("Result with oracle contexts:", evaluate(dev, 'oracle'))
    print("Result with retrieved contexts:", evaluate(dev, 'retrieved', retrieved_documents))