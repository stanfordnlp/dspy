import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import dsp

class TestExtract(unittest.TestCase):
    def setUp(self) -> None:
        self.Question = dsp.Type(
            prefix="Question:", 
            desc="${the question to be answered}")

        self.Answer = dsp.Type(
            prefix="Answer:", 
            desc="${a short factoid answer, often between 1 and 5 words}", 
            format=dsp.format_answers)

        self.Context = dsp.Type(
            prefix="Context:\n",
            desc="${sources that may contain relevant content}",
            format=dsp.passages2text)

        self.Rationale = dsp.Type(
            prefix="Rationale: Let's think step by step.",
            desc="${a step-by-step deduction that identifies the correct response, which will be provided below}"
        )

        self.instructions = "Complete the prompt with short factoid answers. Answer very concisely."

        self.example = dsp.Example(
            id = "42",
            title = "Capital of France",
            context = "France is a country in Europe. Its capital is Paris.",
            question = "What is the capital of France?"
        )

    def test_well_formed1(self):
        template = dsp.Template(
            instructions=self.instructions,
            context=self.Context,
            question=self.Question,
            raionale=self.Rationale,
            answer=self.Answer,
        )

        raw_pred = "Some rationale here. \nAnswer: Frankfurt"
        example = dsp.Example(self.example, answer="Paris")

        completion = template.extract(example, raw_pred, output_fields=["answer"])

        for field in ("id", "title", "context", "question"):
            self.assertEqual(getattr(completion, field), getattr(self.example, field))
        self.assertEqual(completion.answer, "Frankfurt")

    def test_well_formed2(self):
        template = dsp.Template(
            instructions=self.instructions,
            context=self.Context,
            question=self.Question,
            answer=self.Answer,
        )

        raw_pred = "Frankfurt"
        example = dsp.Example(self.example, answer="Paris")

        completion = template.extract(example, raw_pred, output_fields=["answer"])

        for field in ("id", "title", "context", "question"):
            self.assertEqual(getattr(completion, field), getattr(self.example, field))
        self.assertEqual(completion.answer, "Frankfurt")

    def test_malformed(self):
        template = dsp.Template(
            instructions=self.instructions,
            context=self.Context,
            question=self.Question,
            raionale=self.Rationale,
            answer=self.Answer,
        )

        raw_pred = "Some rationale here, and the answer is Frankfurt."
        example = dsp.Example(self.example, answer="Paris")

        completion = template.extract(example, raw_pred, output_fields=["answer"])

        for field in ("id", "title", "context", "question"):
            self.assertEqual(getattr(completion, field), getattr(self.example, field))
        self.assertEqual(completion.answer, "") # Answer should be empty string

if __name__ == "__main__":
    unittest.main()

        