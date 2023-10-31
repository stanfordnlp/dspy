import json
from typing import Any, TypedDict

from datasets import load_dataset


class Example(TypedDict):
    paper_id: str
    example_idx: int
    title: str
    abstract: str
    question: str
    answer: str
    evidence: str


class QASPER:
    def __init__(self):
        self.dataset = load_dataset("allenai/qasper")

    @staticmethod
    def filter_unanswerable(answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [answer_obj for answer_obj in answers if not answer_obj["unanswerable"]]

    @staticmethod
    def get_answer(answer: dict[str, Any]) -> str:
        if answer["yes_no"] != None:
            return str(answer["yes_no"])
        elif answer["free_form_answer"] != "":
            return answer["free_form_answer"]
        else:
            return answer["extractive_spans"][0]

    @staticmethod
    def export_transformed_data(
        data: list[dict[str, Any]], output_file: str = "qasper_processed_dataset.json"
    ) -> None:
        with open(output_file, "w") as file:
            json.dump(data, file, indent=1)

    def transform_for_openqa_rewrite(self) -> None:
        examples: list[Example] = []
        for example_idx, example in enumerate(self.dataset["train"]):
            page_id = example["id"]
            questions = example["qas"]["question"]
            answers = [
                self.filter_unanswerable(answers_list["answer"])
                for answers_list in example["qas"]["answers"]
            ]

            full_text = example["full_text"]

            for question, answer_list in zip(questions, answers):
                if not answer_list:
                    continue

                # each question still has N answers, we pick only one for now
                answer = answer_list[0]
                evidences = answer["evidence"]

                if not (evidences and answer["free_form_answer"]):
                    continue

                evidence_str = ""
                for evidence in evidences:
                    for section, passage in zip(
                        full_text["section_name"], example["full_text"]["paragraphs"]
                    ):
                        if evidence in passage:
                            evidence_str += f"{section}\n-----\n{evidence}\n\n"

                if not evidence_str:
                    continue

                examples.append(
                    Example(
                        paper_id=page_id,
                        example_idx=len(examples),
                        title=example["title"],
                        abstract=example["abstract"],
                        question=question,
                        answer=self.get_answer(answer),
                        evidence=evidence_str,
                    )
                )
        self.export_transformed_data(examples)


if __name__ == "__main__":
    qasper = QASPER()
    qasper.transform_for_openqa_rewrite()
