TASKS = {
    # SQuAD
    "QA": "question"
}


def get_input_ids(tokenizer, input_string):
    return tokenizer(input_string, return_tensors="pt").input_ids


def generate_qa_input(question, context):
    """
    Generate the input for the model to answer the given question
    :param question: Question to answer
    :param context: Context to find answer in
    :return: Input for the QA task
    """
    return f"question: {question}  context: {context}"


def get_task_prefix(task):
    """
    Return the prefix for the given task
    :param task: Task as defined in TASKS
    :return: Full prefix for the task
    """
    if task not in TASKS:
        raise Exception("Specified unknown task")
    return TASKS[task]
