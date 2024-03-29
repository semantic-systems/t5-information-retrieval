# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace NLP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Enron E-Mail Dataset"""

from __future__ import absolute_import, division, print_function

import json
import logging
import os

import nltk

nltk.download('punkt')

import nlp

_CITATION = """\
@article{enrondataset,
       author = {{Enron Corp}, and {Cohen}, William W},
        title = "{Enron Email Dataset}",
      journal = {arXiv e-prints},
         year = 2015,
       urL = " {https://lccn.loc.gov/2018487913}"
}
"""

_DESCRIPTION = """
The original dataset was collected and prepared by the CALO Project \
(A Cognitive Assistant that Learns and Organizes). It contains data from \
about 150 users, mostly senior management of Enron, organized into folders. \
The original corpus contains a total of about 0.5M messages. \
The question and answers were annotated by Alexander Pesterev. The annotated set contains 50 e-mails.
"""

QG_FORMATS = [
    "prepend",
    "highlight",
    "prepend_highlight",
]

NO_ANSWER = {
    "text": "None",
    "answer_start": 0,
    "answer_end": 0
}


class EnronMultitaskConfig(nlp.BuilderConfig):
    """BuilderConfig for Enron."""

    def __init__(self, qg_format="highlight", **kwargs):
        """BuilderConfig for Enron.

    Args:
      **kwargs: keyword arguments forwarded to super.
    """
        super(EnronMultitaskConfig, self).__init__(**kwargs)
        self.qg_format = qg_format


class EnronMultitask(nlp.GeneratorBasedBuilder):
    # TODO: Find a solution to not explicitly write this here
    _URL = "./data/enron_train"
    _DEV_FILE = "valid_output.json"
    _TRAINING_FILE = "train_output.json"

    BUILDER_CONFIGS = [
        EnronMultitaskConfig(
            name=f"{format_}_qg_format",
            version=nlp.Version("1.0.0", "New split API (https://tensorflow.org/datasets/splits)"),
            description="Plain text",
            qg_format=format_
        )
        for format_ in QG_FORMATS
    ]

    def _info(self):
        return nlp.DatasetInfo(
            description=_DESCRIPTION,
            features=nlp.Features(
                {
                    "source_text": nlp.Value("string"),
                    "target_text": nlp.Value("string"),
                    "task": nlp.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        urls_to_download = {
            "train": os.path.join(self._URL, self._TRAINING_FILE),
            "dev": os.path.join(self._URL, self._DEV_FILE),
        }

        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            nlp.SplitGenerator(name=nlp.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            nlp.SplitGenerator(name=nlp.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
        ]

    def _get_correct_alignement(self, context, answer):
        """ Some original examples in SQuAD have indices wrong by 1 or 2 character. We test and fix this here.
        This is not needed to Enron, might add it later though.
        """
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)
        return start_idx, end_idx
        # if context[start_idx:end_idx] == gold_text:
        #     return start_idx, end_idx       # When the gold label position is good
        # elif context[start_idx-1:end_idx-1] == gold_text:
        #     return start_idx-1, end_idx-1   # When the gold label is off by one character
        # elif context[start_idx-2:end_idx-2] == gold_text:
        #     return start_idx-2, end_idx-2   # When the gold label is off by two character
        # else:
        #     raise ValueError()

    def process_qa_text(self, context, question, answer):
        ans_gen_input = f"question: {question}  context: {context}"
        ans_gen_target = f"{answer}"
        return {"source_text": ans_gen_input, "target_text": ans_gen_target, "task": "qa"}

    def process_qg_text(self, context, question, answer):
        answer_text = answer['text'].strip()

        if self.config.qg_format == "prepend":
            que_gen_input = f"answer: {answer_text}  context: {context}"
        elif self.config.qg_format == "highlight":
            start_pos, end_pos = self._get_correct_alignement(context, answer)
            que_gen_input = f"generate question: {context[:start_pos]} {{hl_token}} {answer_text} {{hl_token}} {context[end_pos:]}"
        else:
            start_pos, end_pos = self._get_correct_alignement(context, answer)
            que_gen_input = f"answer: {answer_text} context: {context[:start_pos]} {{hl_token}} {answer_text} {{hl_token}} {context[end_pos:]}"

        que_gen_target = f"{question}"
        return {"source_text": que_gen_input, "target_text": que_gen_target, "task": "qg"}

    def process_e2e_qg(self, paragraph):
        source_text = f"generate questions: {paragraph['context'].strip()}"
        questions = [qas['question'].strip() for qas in paragraph['qas']]
        target_text = " {sep_token} ".join(questions)
        target_text = f"{target_text} {{sep_token}}"
        return {"source_text": source_text, "target_text": target_text, "task": "e2e_qg"}

    def process_ans_ext(self, paragraph):
        context = paragraph['context'].strip()

        # split into sentences
        sents = nltk.sent_tokenize(context)

        # get positions of the sentences
        positions = []
        for i, sent in enumerate(sents):
            if i == 0:
                start, end = 0, len(sent)
            else:
                start, end = (prev_end + 1), (prev_end + len(sent) + 1)
            prev_end = end
            positions.append({'start': start, 'end': end})
        # get answers

        answers = [qa['answers'][0] if not qa["is_impossible"] else NO_ANSWER for qa in paragraph['qas']]

        # get list of answers for each sentence
        sent_answers = []
        for pos, sent in zip(positions, sents):
            target_answers = []
            for ans in answers:
                if ans['answer_start'] in range(pos['start'], pos['end']):
                    target_answers.append(ans['text'].strip())
            sent_answers.append(target_answers)

        # build inputs and targets
        examples = []
        for i, ans in enumerate(sent_answers):
            context = "extract answers:"
            if len(ans) == 0: continue
            ans = list(set(ans))
            for j, sent in enumerate(sents):
                if i == j:
                    sent = "{hl_token} %s {hl_token}" % sent
                context = "%s %s" % (context, sent)
                context = context.strip()
            input_text = context
            target_text = " {sep_token} ".join(ans) + " {sep_token}"

            examples.append({'source_text': input_text, "target_text": target_text, "task": "ans_ext"})

        return examples

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logging.info("generating examples from = %s", filepath)
        count = 0
        tasks = ['qa', 'qg', 'ans_ext', 'e2e_qg']
        with open(filepath) as f:
            squad = json.load(f)
            for article in squad["data"]:
                title = article.get("title", "").strip()
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"].strip()

                    if 'ans_ext' in tasks:
                        ans_ext_examples = self.process_ans_ext(paragraph)
                        for example in ans_ext_examples:
                            yield count, example
                            count += 1

                    if 'e2e_qg' in tasks:
                        yield count, self.process_e2e_qg(paragraph)
                        count += 1

                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        id_ = qa["id"]

                        answers = [answer["text"].strip() for answer in qa["answers"]]

                        for task in tasks:
                            if task == 'qa':
                                if len(answers) == 0:
                                    yield count, self.process_qa_text(context, question, NO_ANSWER["text"])
                                    count += 1
                                else:
                                    for answer in answers:
                                        yield count, self.process_qa_text(context, question, answer)
                                        count += 1
                            if task == 'qg':
                                if len(qa["answers"]) == 0:
                                    qa["answers"].append(NO_ANSWER)
                                yield count, self.process_qg_text(context, question, qa["answers"][0])
                                count += 1
