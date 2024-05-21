from typing import Any

from tests_integration.base import BaseIntegrationTestWithCache


class TestIntroIntegration(BaseIntegrationTestWithCache):
    def test_dspy_workflow(self) -> None:
        dspy = self.setup_dspy()

        dev_example, dev_set, training_set = self.assert_dataset_loading()

        self.assert_basic_qa(dev_example, dspy)

        self.assert_retrieval(dev_example, dspy)

        self.assert_compilation(dev_set, dspy, training_set)

    def assert_compilation(self, devset, dspy, trainset) -> None:
        class GenerateAnswer(dspy.Signature):
            """Answer questions with short factoid answers."""

            context = dspy.InputField(desc="may contain relevant facts")
            question = dspy.InputField()
            answer = dspy.OutputField(desc="often between 1 and 5 words")

        class RAG(dspy.Module):
            def __init__(self, num_passages=3):
                super().__init__()

                self.retrieve = dspy.Retrieve(k=num_passages)
                self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

            def forward(self, question):
                context = self.retrieve(question).passages
                prediction = self.generate_answer(context=context, question=question)
                return dspy.Prediction(context=context, answer=prediction.answer)

        from dspy.teleprompt import BootstrapFewShot

        # Validation logic: check that the predicted answer is correct.
        # Also check that the retrieved context actually contains that answer.
        def validate_context_and_answer(example, pred, trace=None):  # noqa
            answer_em = dspy.evaluate.answer_exact_match(example, pred)
            answer_pm = dspy.evaluate.answer_passage_match(example, pred)
            return answer_em and answer_pm

        # Set up a basic teleprompter, which will compile our RAG program.
        teleprompter = BootstrapFewShot(metric=validate_context_and_answer)
        # Compile the RAG model
        compiled_rag = teleprompter.compile(RAG(), trainset=trainset)

        # Test the compiled RAG model with a question
        my_question = "What castle did David Gregory inherit?"
        pred = compiled_rag(my_question)

        # Assertions to verify the compiled RAG model
        assert f"Question: {my_question}" == "Question: What castle did David Gregory inherit?"
        assert f"Predicted Answer: {pred.answer}" == "Predicted Answer: Kinnairdy Castle"
        assert f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}" == (
            "Retrieved Contexts (truncated): ['David Gregory (physician) | David Gregory (20 December 1625 – 1720) "
            "was a Scottish physician and inventor."
            "His surname is sometimes spelt as Gregorie, the original Scottish spelling. He inherited Kinn...', "
            "'Gregory Tarchaneiotes | Gregory Tarchaneiotes (Greek: Γρηγόριος Ταρχανειώτης , Italian: \"Gregorio "
            'Tracanioto" or "Tracamoto" ) was a "protospatharius" and the long-reigning catepan of Italy from 998 '
            "t...', "
            "'David Gregory (mathematician) | David Gregory (originally spelt Gregorie) FRS (? 1659 – 10 October "
            "1708) was a Scottish mathematician and astronomer."
            "He was professor of mathematics at the University ...']"
        )

        # Verify compiled model's parameters
        for name, parameter in compiled_rag.named_predictors():
            assert name is not None
            assert parameter.demos[0] is not None

        from dspy.evaluate.evaluate import Evaluate

        # Set up the evaluation function
        evaluate_on_hotpotqa = Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=5)
        # Evaluate the compiled RAG program with the exact match metric
        metric = dspy.evaluate.answer_exact_match
        evaluate_on_hotpotqa(compiled_rag, metric=metric)

        def gold_passages_retrieved(example, pred, trace=None):  # noqa
            gold_titles = set(map(dspy.evaluate.normalize_text, example["gold_titles"]))
            found_titles = set(map(dspy.evaluate.normalize_text, [c.split(" | ")[0] for c in pred.context]))
            return gold_titles.issubset(found_titles)

        compiled_rag_retrieval_score = evaluate_on_hotpotqa(compiled_rag, metric=gold_passages_retrieved)

        class GenerateSearchQuery(dspy.Signature):
            """Write a simple search query that will help answer a complex question."""

            context = dspy.InputField(desc="may contain relevant facts")
            question = dspy.InputField()
            query = dspy.OutputField()

        from dsp.utils import deduplicate

        class SimplifiedBaleen(dspy.Module):
            def __init__(self, passages_per_hop=3, max_hops=2):
                super().__init__()

                self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
                self.retrieve = dspy.Retrieve(k=passages_per_hop)
                self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
                self.max_hops = max_hops

            def forward(self, question):
                context = []

                for hop in range(self.max_hops):
                    query = self.generate_query[hop](context=context, question=question).query
                    passages = self.retrieve(query).passages
                    context = deduplicate(context + passages)

                pred = self.generate_answer(context=context, question=question)
                return dspy.Prediction(context=context, answer=pred.answer)

        # Test the SimplifiedBaleen model with a question
        my_question = "How many storeys are in the castle that David Gregory inherited?"
        uncompiled_baleen = SimplifiedBaleen()
        pred = uncompiled_baleen(my_question)

        # Assertions to verify the SimplifiedBaleen model
        assert (
            f"Question: {my_question}" == "Question: How many storeys are in the castle that David Gregory inherited?"
        )
        assert f"Predicted Answer: {pred.answer}" == "Predicted Answer: five"
        assert f"Retrieved Contexts (truncated): {[c[:20] + '...' for c in pred.context]}" == (
            "Retrieved Contexts (truncated): ['David Gregory (physi...', 'The Boleyn Inheritan...', 'Gregory of Gaeta "
            "| G...',"
            "'Kinnairdy Castle | K...', 'Kinnaird Head | Kinn...', 'Kinnaird Castle, Bre...']"
        )

        def validate_context_and_answer_and_hops(example, pred, trace=None):
            if not dspy.evaluate.answer_exact_match(example, pred):
                return False
            if not dspy.evaluate.answer_passage_match(example, pred):
                return False

            hops = [example.question] + [outputs.query for *_, outputs in trace if "query" in outputs]

            if max([len(h) for h in hops]) > 100:
                return False
            if any(
                dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))
            ):
                return False

            return True

        teleprompter = BootstrapFewShot(metric=validate_context_and_answer_and_hops)
        compiled_baleen = teleprompter.compile(
            SimplifiedBaleen(),
            teacher=SimplifiedBaleen(passages_per_hop=2),
            trainset=trainset,
        )
        uncompiled_baleen_retrieval_score = evaluate_on_hotpotqa(uncompiled_baleen, metric=gold_passages_retrieved)
        compiled_baleen_retrieval_score = evaluate_on_hotpotqa(compiled_baleen, metric=gold_passages_retrieved)

        # Assertions for the retrieval scores
        assert f"## Retrieval Score for RAG: {compiled_rag_retrieval_score}" == "## Retrieval Score for RAG: 26.0"
        assert (
            f"## Retrieval Score for uncompiled Baleen: {uncompiled_baleen_retrieval_score}"
            == "## Retrieval Score for uncompiled Baleen: 36.0"
        )
        assert (
            f"## Retrieval Score for compiled Baleen: {compiled_baleen_retrieval_score}"
            == "## Retrieval Score for compiled Baleen: 60.0"
        )
        assert compiled_baleen("How many storeys are in the castle that David Gregory inherited?") is not None

    def assert_retrieval(self, dev_example, dspy) -> None:
        retrieve = dspy.Retrieve(k=3)
        top_k_passages = retrieve(dev_example.question).passages

        # Assertions to verify the retrieval functionality
        assert retrieve.k == 3
        assert (
            dev_example.question
            == "What is the nationality of the chef and restaurateur featured in Restaurant: Impossible?"
        )
        assert (
            top_k_passages[0]
            == "Restaurant: Impossible | Restaurant: Impossible is an American reality television series, featuring "
            "chef and restaurateur Robert Irvine, that aired on Food Network from 2011 to 2016."
        )
        assert (
            top_k_passages[1]
            == "Jean Joho | Jean Joho is a French-American chef and restaurateur. He is chef/proprietor of Everest in "
            "Chicago (founded in 1986), Paris Club Bistro & Bar and Studio Paris in Chicago, The Eiffel Tower "
            "Restaurant in Las Vegas, and Brasserie JO in Boston."
        )
        assert top_k_passages[2] == (
            "List of Restaurant: Impossible episodes | This is the list of the episodes for the American cooking and "
            'reality television series "Restaurant Impossible", '
            "produced by Food Network. The premise of the series is that within two days and on a budget of $10,000, "
            "celebrity chef Robert Irvine renovates a failing American restaurant with the goal of helping to restore "
            "it to profitability and prominence."
            "Irvine is assisted by a designer (usually Taniya Nayak, Cheryl Torrenueva, or Lynn Keagan, but sometimes "
            "Vanessa De Leon, Krista Watterworth, Yvette Irene, or Nicole Faccuito), along with general contractor "
            "Tom Bury, who sometimes does double duty as both general contractor and designer."
            "After assessing the problems with the restaurant, Robert Irvine typically creates a plan for the new "
            "decor, oversees the cleaning of the restaurant, reduces the size of the menu and improves the food, "
            "develops a promotional activity, educates the restaurant's owners, or trains the staff, as needed by "
            "each restaurant."
        )

        retrieved_value = retrieve("When was the first FIFA World Cup held?").passages[0]
        assert retrieved_value == (
            "History of the FIFA World Cup | The FIFA World Cup was first held in 1930, when FIFA president Jules "
            "Rimet decided to stage an international football tournament."
            "The inaugural edition, held in 1930, was contested as a final tournament of only thirteen teams invited "
            "by the organization."
            "Since then, the World Cup has experienced successive expansions and format remodeling to its current "
            "32-team final tournament preceded by a two-year qualifying process, involving over 200 teams from around "
            "the world."
        )

    def assert_basic_qa(self, dev_example, dspy) -> None:
        class BasicQA(dspy.Signature):
            """Answer questions with short factoid answers."""

            question = dspy.InputField()
            answer = dspy.OutputField(desc="often between 1 and 5 words")

        # Define the predictor
        generate_answer = dspy.Predict(BasicQA)
        # Call the predictor on a particular input
        pred = generate_answer(question=dev_example.question)

        # Assertions to verify the basic QA functionality
        assert (
            f"Question: {dev_example.question}"
            == "Question: What is the nationality of the chef and restaurateur featured in Restaurant: Impossible?"
        )
        assert f"Predicted Answer: {pred.answer}" == "Predicted Answer: American"

        # Define the predictor with chain of thought
        generate_answer_with_chain_of_thought = dspy.ChainOfThought(BasicQA)
        # Call the predictor on the same input
        pred = generate_answer_with_chain_of_thought(question=dev_example.question)

        # Assertions to verify the chain of thought functionality
        assert (
            f"Question: {dev_example.question}"
            == "Question: What is the nationality of the chef and restaurateur featured in Restaurant: Impossible?"
        )
        assert (
            f"Thought: {pred.rationale.split('.', 1)[1].strip()}"
            == "Thought: We know that the chef and restaurateur featured in Restaurant: Impossible is Robert Irvine."
        )
        assert f"Predicted Answer: {pred.answer}" == "Predicted Answer: British"

    def assert_dataset_loading(self) -> None:
        from dspy.datasets import HotPotQA

        # Load the dataset
        dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)

        # Prepare the datasets for training and development
        trainset = [x.with_inputs("question") for x in dataset.train]
        devset = [x.with_inputs("question") for x in dataset.dev]
        train_example = trainset[0]

        # Assertions to verify the dataset loading
        assert (
            f"Question: {train_example.question}"
            == "Question: At My Window was released by which American singer-songwriter?"
        )
        assert f"Answer: {train_example.answer}" == "Answer: John Townes Van Zandt"
        dev_example = devset[18]
        assert (
            f"Question: {dev_example.question}"
            == "Question: What is the nationality of the chef and restaurateur featured in Restaurant: Impossible?"
        )
        assert f"Answer: {dev_example.answer}" == "Answer: English"
        assert "Restaurant: Impossible" in list(dev_example.gold_titles)
        assert "Robert Irvine" in list(dev_example.gold_titles)
        assert (
            f"For this dataset, training examples have input keys {train_example.inputs().keys()} and label keys "
            f"{train_example.labels().keys()}"
            == "For this dataset, training examples have input keys ['question'] and label keys ['answer']"
        )
        assert (
            f"For this dataset, dev examples have input keys {dev_example.inputs().keys()} and label keys "
            f"{dev_example.labels().keys()}"
            == "For this dataset, dev examples have input keys ['question'] and label keys ['answer', 'gold_titles']"
        )
        return dev_example, devset, trainset

    def setup_dspy(self) -> Any:
        import dspy

        turbo = dspy.OpenAI(model="gpt-3.5-turbo")
        colbertv2_wiki17_abstracts = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
        dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)
        return dspy
