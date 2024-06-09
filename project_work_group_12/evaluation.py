from sentence_transformers import SentenceTransformer

from sentence_transformers import SentenceTransformer, util
from working_file import get_relevant_documents
from evaluate import load
from gfw import DevDataset, GPTQA
import json

def bertscore(correct_answers, predicted_answers):
    # Load the pre-trained Sentence-BERT model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Check if the input lists have the same length
    if len(correct_answers) != len(predicted_answers):
        raise ValueError("The lists of correct and predicted answers must have the same length")

    # Initialize a list to store similarity scores
    similarity_scores = []

    # Compute the similarity score for each pair of answers
    for correct, predicted in zip(correct_answers, predicted_answers):
        # Encode the sentences to get their embeddings
        embedding1 = model.encode(correct, convert_to_tensor=True)
        embedding2 = model.encode(predicted, convert_to_tensor=True)

        # Compute the cosine similarity between the embeddings
        similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()
        similarity_scores.append(similarity_score)

    # Calculate the average similarity score
    average_score = sum(similarity_scores) / len(similarity_scores)

    return average_score

def exact_match(predictions: list[str], references: list[str]) -> list[float]:
    """
    """
    exact_match_metric = load("exact_match")
    results = exact_match_metric.compute(predictions=predictions,
                                         references=references,
                                         ignore_case=True,
                                         ignore_punctuation=True)

    return results['exact_match']

def evaluate(dev: DevDataset, mode: str = 'oracle', retrieved_contexts: dict =None, write_to_file: tuple[bool, str] = (False, 'out.txt'), limit: int = 10):
    ground_truths = []
    answers = []
    prompts = []

    gpt = GPTQA()

    count = 0
    for question in dev:
        if count is not None:
            if count > limit: break
        if mode == 'oracle':
            context = question.get_contexts('gold')
        elif mode == 'retrieved' and retrieved_contexts is not None:
            context = retrieved_contexts.get(question.id)
        else: raise ValueError("Something wrong happened")

        response = gpt.ask_question(question, context=context)
        ground_truths.append(question.answer)
        answers.append(response[1])
        prompts.append(response[0])

        count += 1

    print(answers)
    print(ground_truths)

    if write_to_file[0]:
        with open(write_to_file[1], 'w', encoding='utf-8') as f:
            for prompt, response, ground_truth in zip(prompts, answers, ground_truths):
                f.write(f"_____Prompt_____\n{prompt}\n\n_____Response_____\n{response}\n\n_____Ground Truth_____\n{ground_truth}\n\n\n")
                
    return exact_match(answers, ground_truths)

def evaluate_retrieved_documents(dev, retrieved_contexts: dict, K: list[int], write_to_file: bool = True, limit: int = 10, metric = 'ExactMatch'):
    """
    Evaluated performance over retrieved documents using the top-k contained in list K.
    :param retrieved_contexts:
    :param K:
    :return:
    """
    responses = {k: ([], []) for k in K}
    metrics = {}

    for k in K:
        progress = 1
        for question_id, context in retrieved_contexts.items():
            if limit is not None:
                if progress > 10: break
            question = dev[question_id]
            context = [c.replace('\n\n', '\n') for c in context]
            response = gpt.ask_question(question, context[0:k])
            responses[k][0].append(response[1])
            responses[k][1].append(response[2])

            if progress % 10 == 0: print(f'k={k}: {round(progress / len(retrieved_contexts) * 100, 1)}%')
            progress += 1
        if metric == 'ExactMatch':
            metrics[k] = exact_match(responses[k][0], responses[k][1])
        elif metric == 'BertMatch':
            print('risposta 1: ', responses[k][0],' risposta 2: ', responses[k][1])
            metrics[k] = bertscore(responses[k][0], responses[k][1])


    if write_to_file:
        with open('results_retrieved.json', 'w', encoding='utf-8') as f:
            json.dump(responses, f, ensure_ascii=False, indent=4)

    return metrics

if __name__ == "__main__":
    dev = DevDataset(r"C:\Users\User\Downloads\dev_full.json")

    gpt = GPTQA()
    gpt.set_system_prompt("For this task, you should provide a factoid answer. This means that you are limited to returning the exact answer to the question"
                          "(which might be a person's name, a date, a place and so on), without any additional word and without putting the factoid answer in"
                          "a sentence. For instance, if the question is \"Who was the lead singer of the rock band Queen?\", you should reply \"Freddy Mercury\","
                          "and not \"The lead singer of the band Queen was Freddy Mercury\".", )

    # *** EVALUATE RAG WITH RETRIEVED DOCUMENTS (FOR TOP_K = 1,3,5) ***

    top_k = 10
    path = r"C:\Users\User\Downloads\results_10.json"
    with open(path, 'r', encoding='cp850') as file: # but i'm not sure if it really uploads stuff as it should....
        retrieved_documents = json.load(file)

    result = evaluate_retrieved_documents(dev, retrieved_documents, [1,3,5], metric = 'BertMatch')

    # *** EVALUATION FOR ORACLE CONTEXTS ***
    # i don't find the file where it saves stuff

    print("Result with oracle contexts:", evaluate(dev, 'oracle', write_to_file=(True, "results_oracle.json")))