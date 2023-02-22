"""
This script is used for different preprocessing steps needed for data coming in SQUAD format
- Combines answers to similar (identical) questions into one single question object
- Removes \n tags in answers
"""
import json


def main():
    process_file("train_output")
    process_file("valid_output")


def process_file(file):
    """
    Reads the questions from the file and processes them
    """
    qas = load_qas(f"{file}.json")
    # paragraph is equal to a document, since we only have one paragraph per document
    for document in qas:
        questions = document["paragraphs"][0]["qas"]
        # combine the answers per unique question
        unique_questions = combine_into_unique_questions(questions)
        document["paragraphs"][0]["qas"] = list(unique_questions.values())
    with open(f"{file}_preprocessed.json", 'w') as result_file:
        write_object = {
            "data": qas
        }
        result_file.write(json.dumps(write_object, indent=2))


def combine_into_unique_questions(questions):
    unique_questions = {}
    for question in questions:
        question = remove_special_chars_from_answers(question)
        if question["question"] not in unique_questions.keys():
            unique_questions[question["question"]] = question
        else:
            unique_questions[question["question"]]["answers"] += question["answers"]
    return unique_questions


def remove_special_chars_from_answers(question):
    """
    Removes any special characters like \n from the answers of the question
    :param question: Question object
    :return: \n removed from answers
    """
    for answer in question["answers"]:
        answer["text"] = answer["text"].replace("\n", "").strip()
    return question


def load_qas(file, limit=999999, offset=0):
    with open(file) as file:
        json_file = json.load(file)
        return json_file["data"][offset:limit]


if __name__ == "__main__":
    main()
