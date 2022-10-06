from pipelines import pipeline
import sys

nlp = None


def extract_question_and_answer(extract_from):
    return nlp(extract_from)


def get_answer_for_question(extract_from, question):
    return nlp({
        "question": question,
        "context": extract_from
    })


def load_test_data(file):
    file = open('./data/enron/' + file, "r", encoding="utf-8")
    return file.read()


def get_pipeline(arguments):
    global nlp
    if len(arguments) == 3:
        nlp = pipeline("multitask-qa-qg", model="valhalla/t5-base-qg-hl")
    else:
        # nlp = pipeline("question-generation", model="valhalla/t5-base-qg-hl")
        nlp = pipeline("question-generation")


# python enron.py "83_"
if __name__ == '__main__':
    text = load_test_data(sys.argv[1])
    get_pipeline(sys.argv)

    if len(sys.argv) == 3:
        question_arg = sys.argv[2]
        print(get_answer_for_question(text, question_arg))
    else:
        print(extract_question_and_answer(text))
