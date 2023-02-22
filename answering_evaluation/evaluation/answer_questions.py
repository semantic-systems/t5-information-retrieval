from Pipeline import Pipeline
import json
from helpers import categorize_question

DATASET_PATH = "../"
DATASET_NAME = "valid_output_preprocessed"
MODEL_TO_USE = "flan-t5-base"
# MODEL_FILE_NAME = MODEL_TO_USE.strip("google/")
MODEL_FILE_NAME = MODEL_TO_USE
IS_FINE_TUNED = True
WITH_NONE = False
MAX_LEN = 4096


def main():
    print(f"Starting answering pipeline for model {MODEL_TO_USE} (fine_tuned={IS_FINE_TUNED}, with_none={WITH_NONE})")
    model = Pipeline(MODEL_TO_USE, is_fine_tuned=IS_FINE_TUNED, model_max_length=MAX_LEN)
    answers = answer_questions(DATASET_PATH + DATASET_NAME + ".json", model, with_none=WITH_NONE)
    file_ending = "_without_" if not WITH_NONE else "_with_none.json"
    file_name = DATASET_NAME + "_" + MODEL_FILE_NAME + f"_fine_tuned_model_answers{file_ending}" if IS_FINE_TUNED \
        else DATASET_NAME + "_" + MODEL_FILE_NAME + f"_model_answers{file_ending}"
    with open(file_name, "w") as file:
        file.write(json.dumps(answers, indent=2))
    # print(json.dumps(answers, indent=2))


def answer_questions(path, model, doc_limit=0, with_none=True):
    """
    :param path: Path to the qa data
    :param model: model to use for answer
    :param doc_limit: If greater than 0: Limit of documents to load
    :param with_none: Ask the model for an answer if none is present?
    :return:
    """
    qas = load_qas(path, doc_limit) if doc_limit > 0 else load_qas(path)
    for count, context in enumerate(qas):
        for question in context["questions"]:
            # If the question has no answers, append a None answer for later use
            if with_none:
                if len(question["answers"]) == 0:
                    question["answers"].append({
                        "answer_start": 0,
                        "text": "None",
                    })
            question["model_answers"] = model.answer_question(question["question"], context["context"],
                                                              num_answers=len(question["answers"]))
        print(f"Answered {count + 1} of {len(qas)} documents")
    return qas


def load_qas(path, limit=999999):
    """
    :param path:
    :param limit:
    :return: {
        context: String,
        questions: [{
                question: String,
                answers: [String]
            }]
    }
    """
    qas = []
    with open(path) as file:
        json_file = json.load(file)
        for paragraph in json_file["data"][:limit]:
            for qa in paragraph["paragraphs"]:
                context = {
                    "context": qa["context"],
                    "questions": []
                }
                for question in qa["qas"]:
                    question_text = question["question"]
                    answers = []
                    for answer in question["answers"]:
                        answers.append(answer)
                    context["questions"].append({
                        "question": question_text,
                        "question_category": categorize_question(question_text),
                        "answers": answers
                    })
                qas.append(context)
    return qas


if __name__ == "__main__":
    main()
