import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration, AutoTokenizer
import tokenizer_helpers


class Pipeline:
    """
    Pipleine for using a T5 model for question answering
    """
    VALID_MODELS = ["t5-small", "t5-base", "t5-large", "google/t5-small-ssm-nq", "google/t5-base-ssm-nq",
                    "google/t5-large-ssm-nq", "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large"]

    def __init__(self, model, is_fine_tuned, model_max_length=1024, use_cuda=False):
        # if model not in self.VALID_MODELS:
        #     raise ValueError("Specified model is not supported")
        self.model_name = model if not is_fine_tuned else f"{model}_fine_tuned"
        self.tokenizer = get_tokenizer(model, model_max_length=model_max_length)
        self.model = get_model(model, is_fine_tuned)

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)

    def answer_question(self, question, context, num_answers=1):
        if num_answers == 0:
            return []
        prompt = tokenizer_helpers.generate_qa_input(question, context)

        features = self.tokenizer(
            prompt,
            padding="longest",
            max_length=len(prompt),
            truncation=True,
            return_tensors="pt"
        )

        model_output = self.model.generate(
            input_ids=features["input_ids"].to(self.device),
            attention_mask=features['attention_mask'].to(self.device),
            max_new_tokens=len(context),
            num_beams=num_answers,
            do_sample=False,
            # num_beams=4,
            # length_penalty=0.6,
            num_return_sequences=num_answers
        )

        return [self.tokenizer.decode(out, skip_special_tokens=True) for out in model_output]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model_name})"


def get_tokenizer(model, model_max_length=512):
    print("get tokenizer for", model)
    if "ssm-nq" in model:
        # return AutoTokenizer.from_pretrained("google/t5-small-ssm-nq", model_max_length=model_max_length)
        return AutoTokenizer.from_pretrained("t5-base", model_max_length=model_max_length)
    elif "flan-t5" in model:
        return T5TokenizerFast.from_pretrained("google/flan-t5-base", model_max_length=model_max_length)
    return T5TokenizerFast.from_pretrained(model, model_max_length=model_max_length)


def get_model(model, is_fine_tuned):
    if is_fine_tuned:
        print("Loading fine tuned model from", f"../trained models/{model}-trained")
        return T5ForConditionalGeneration.from_pretrained(f"../trained models/{model}-trained")
    return T5ForConditionalGeneration.from_pretrained(model)
