import json
import re
import string
from collections import Counter
from helpers import QuestionType, categorize_question

import evaluate

MODEL_TO_CHECK = "t5-large"
# MODEL_FILE_NAME = MODEL_TO_CHECK.strip("google/")
MODEL_FILE_NAME = MODEL_TO_CHECK
IS_FINE_TUNED = True
WITH_NONE = False
EVALUATE_PER_QUESTION_TYPE = True
EVALUATE_STRATEGIES = True


def evaluate_all():
    file_ending = "_without_none" if not WITH_NONE else "_with_none"
    file_ending += ".json"
    file_name = f"./model_answers/valid_output_preprocessed_{MODEL_FILE_NAME}_model_answers{file_ending}" if not IS_FINE_TUNED else \
        f"./model_answers/valid_output_preprocessed_{MODEL_FILE_NAME}_fine_tuned_model_answers{file_ending}"
    with open(file_name) as file:
        predictions, references = convert_answers_doc(json.load(file))
    squad_metric = evaluate.load("squad_v2")
    results = squad_metric.compute(predictions=predictions, references=references)
    res_file_name = f"{MODEL_FILE_NAME}_evaluation{file_ending}" if not IS_FINE_TUNED else \
        f"{MODEL_FILE_NAME}_fine_tuned_evaluation{file_ending}"
    with open(res_file_name, 'w') as result_file:
        result_file.write(json.dumps(results, indent=2))
    # print(results)


def evaluate_per_question_type():
    squad_metric = evaluate.load("squad_v2")
    file_ending = "_without_none" if not WITH_NONE else "_with_none"
    file_ending += ".json"
    file_name = f"./model_answers/valid_output_preprocessed_{MODEL_FILE_NAME}_model_answers{file_ending}" if not IS_FINE_TUNED else \
        f"./model_answers/valid_output_preprocessed_{MODEL_FILE_NAME}_fine_tuned_model_answers{file_ending}"
    results = {}
    json_file = None
    with open(file_name) as file:
        json_file = json.load(file)
    for question_type in QuestionType:
        predictions, references = convert_answers_doc(json_file, only_question_type=question_type)
        results[question_type] = squad_metric.compute(predictions=predictions, references=references)
    res_file_name = f"{MODEL_FILE_NAME}_evaluation_per_question_type{file_ending}" if not IS_FINE_TUNED else \
        f"{MODEL_FILE_NAME}_fine_tuned_evaluation_per_question_type{file_ending}"
    with open(res_file_name, 'w') as result_file:
        result_file.write(json.dumps(results, indent=2))
    # print(results)


def evaluate_strategies():
    STRATEGIES = ["beam_search_greedy", "beam_search_sample", "contrastive_search", "diverse_beam_search",
                  "multinomial_sampling"]
    squad_metric = evaluate.load("squad_v2")
    results = {}
    for strategy in STRATEGIES:
        with open("./results/per_strategy/" + strategy + ".json") as file:
            json_file = json.load(file)
        predictions, references = convert_answers_doc(json_file)
        results[strategy] = squad_metric.compute(predictions=predictions, references=references)
    file_name_suffix = "_per_question_type.json" if EVALUATE_PER_QUESTION_TYPE else ".json"
    with open("strategy_evaluation" + file_name_suffix, 'w') as result_file:
        result_file.write(json.dumps(results, indent=2))


def evaluate_strategies_per_question_type():
    STRATEGIES = ["beam_search_greedy", "beam_search_sample", "contrastive_search", "diverse_beam_search",
                  "multinomial_sampling"]
    data = {}
    for strategy in STRATEGIES:
        with open("./results/per_strategy/" + strategy + ".json") as file:
            data[strategy] = json.load(file)
    squad_metric = evaluate.load("squad_v2")
    results = {}
    for question_type in QuestionType:
        results[question_type] = {
            "strategies": [],
            "EM": [],
            "F1": []
        }
        for strategy in STRATEGIES:
            predictions, references = convert_answers_doc(data[strategy], only_question_type=question_type)
            squad_results = squad_metric.compute(predictions=predictions, references=references)
            results[question_type]["strategies"].append(strategy)
            results[question_type]["EM"].append(squad_results["exact"])
            results[question_type]["F1"].append(squad_results["f1"])
    with open("strategy_evaluation_per_question_type_no_none.json", 'w') as result_file:
        result_file.write(json.dumps(results, indent=2))


def convert_answers_doc(json_doc, only_question_type=None, doc_limit=9999999):
    """
    Converts the model answers document as generated from the answers_questions.py script to two files containing
    the answers and the predictions as described in https://huggingface.co/spaces/evaluate-metric/squad_v2
    :return: predictions, references
    """
    qa_id = 0
    predictions, references = [], []
    for document in json_doc[:doc_limit]:
        for question in document["questions"]:
            if only_question_type and categorize_question(question["question_category"]) != only_question_type:
                continue
            for idx, answer in enumerate(question["answers"]):
                if not WITH_NONE and answer["text"] == "None":
                    continue
                references.append(
                    {"answers": {
                        "answer_start": [answer["answer_start"]],
                        "text": [answer["text"]]
                    },
                        "id": str(qa_id)}
                )

                if len(question["model_answers"]) > 1:
                    prediction, new_predictions_list = get_highest_matching(answer["text"], question["model_answers"])
                    question["model_answers"] = new_predictions_list
                else:
                    prediction = question["model_answers"][0]

                predictions.append({
                    "id": str(qa_id),
                    "prediction_text": prediction,
                    "no_answer_probability": 0.
                })
                qa_id += 1
    # print("references", references)
    # print("predictions", predictions)
    return predictions, references


def get_highest_matching(reference, predictions, debug=False):
    f"""
    Takes as an input a reference and a list of predictions.Retrieves the highest matching prediction.
    Then removes that prediction from the list and returns the prediction with the new list.
    :param reference: {str}
    :param predictions: {str}
    :param debug: {bool} active some additional printing of infos
    :return:
    """
    if debug:
        print(f"finding highest matching prediction for {reference} among {predictions}")
    highest = 0
    prediction_index = 0
    for idx, p in enumerate(predictions):
        f1 = f1_score(reference, p)
        if f1 > highest:
            highest = f1
            prediction_index = idx
    to_return = predictions[prediction_index]
    del predictions[prediction_index]
    if debug:
        print(f"returning {to_return}")
    return to_return, predictions


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


if __name__ == '__main__':
    if EVALUATE_PER_QUESTION_TYPE and not EVALUATE_STRATEGIES:
        evaluate_per_question_type()
    elif EVALUATE_STRATEGIES:
        evaluate_strategies() if not EVALUATE_PER_QUESTION_TYPE else evaluate_strategies_per_question_type()
    else:
        evaluate_all()
