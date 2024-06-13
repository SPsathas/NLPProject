import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from evaluate import load
from metrics.ExactMatch import ExactMatch


def compute_cosine_similarity(sentence1, sentence2):
    """
    Compute the cosine similarity between two sentences using TF-IDF.

    Args:
    sentence1 (str): The first sentence.
    sentence2 (str): The second sentence.

    Returns:
    float: The cosine similarity score between the two sentences.
    """

    # Create a TF-IDF vectorizer

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Transform the sentences into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])

    # Compute the cosine similarity between the sentences
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    # Print the similarity score
    similarity_score = similarity_matrix[0][0]

    return similarity_score


def exact_match(prediction, reference) -> float:
    """
    """
    exact_match_metric = ExactMatch()
    results = exact_match_metric.evaluate(prediction, reference)

    return results



def scores(responde_file, K):
    """
    Calculate the BERT similarity scores between correct and predicted answers,
    and create a new CSV file with these scores.

    :param question: The list of questions corresponding to the answers
    :param correct_answers: A list of correct answers
    :param predicted_answers: A list of predicted answers
    :param evaluation_csv_path: Path to the CSV file to be created
    :param K: Column name for the similarity scores in the CSV
    """
    questions = responde_file['Prompt']
    predicted_answers = responde_file['Response']
    correct_answers = responde_file['Ground Truth']
    # Load the pre-trained Sentence-BERT model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Check if the input lists have the same length
    if len(correct_answers) != len(predicted_answers):
        raise ValueError("The lists of correct and predicted answers must have the same length")

    # Prepare data for the DataFrame
    data_list = []
    index = 0
    # Compute the similarity score for each pair of answers
    for correct, predicted, question in zip(correct_answers, predicted_answers, questions):
        # Encode the sentences to get their embeddings
        embedding1 = model.encode(correct, convert_to_tensor=True)
        embedding2 = model.encode(predicted, convert_to_tensor=True)

        # Compute the cosine similarity between the embeddings
        bert_score = util.pytorch_cos_sim(embedding1, embedding2).item()
        cosine_score = compute_cosine_similarity(correct, predicted)
        exact_match_score = exact_match(correct, predicted)
        # Collect data
        data_list.append({'index': index,
                          'question': question,
                          'predicted_answer': predicted,
                          'actual_answer': correct,
                          'bert_score': bert_score,
                          'cosine_score': cosine_score,
                          'exact_match': exact_match_score
                          })
        index += 1
    # Create a DataFrame from collected data
    data = pd.DataFrame(data_list)

    # Save the DataFrame to a new CSV file
    return data


def calculate_bert_scores(file_path1, file_path2):
    """
    Calculate the BERT scores between corresponding predicted answers in two result files.

    :param file_path1: Path to the first CSV file (results_retrival_1.csv).
    :param file_path2: Path to the second CSV file (results_retrival_3.csv).
    :return: DataFrame with the BERT scores for corresponding rows.
    """
    # Load the CSV files
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)

    # Load the pre-trained Sentence-BERT model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Check if the files have the same number of rows
    if len(df1) != len(df2):
        raise ValueError("Both files must have the same number of rows for 1-to-1 comparison.")

    # List to hold the score data
    scores_list = []

    # Iterate over the rows in both DataFrames by index
    for index, (predicted1, predicted2) in enumerate(zip(df1['predicted_answer'], df2['predicted_answer'])):
        # Encode the predicted answers to get their embeddings
        embedding1 = model.encode(predicted1, convert_to_tensor=True)
        embedding2 = model.encode(predicted2, convert_to_tensor=True)

        # Compute the cosine similarity between the embeddings
        bert_score = util.pytorch_cos_sim(embedding1, embedding2).item()

        # Append the results to the list
        scores_list.append({
            'Index': index,
            'Predicted_Answer_File1': predicted1,
            'Predicted_Answer_File2': predicted2,
            'BERT_Score': bert_score
        })

    # Create a DataFrame from collected data
    result_df = pd.DataFrame(scores_list)
    result_df.to_csv(f'data/results_cross_scores.csv', index=False)

# for k in [0, 1, 3, 5]:
#     file = pd.read_csv(f'data/k{k}.csv', sep=';')
#     data = scores(file, k)
#     data.to_csv(f'data/results_retrival_{k}.csv', index=False)
#     print(f'FINISHED WITH k{k}')

