import os
import pickle
import numpy as np
import openai
import pandas as pd
import tqdm
from IPython.display import display

import dsp
from experiments.reranker.hotpot_qa_program import multihop_QA

os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(".", "cache")

import ujson

from experiments.reranker.reranker import (
    SentenceTransformersCrossEncoder,
    retrieveRerank,
    retrieveRerankEnsemble,
    retrieveRerankEnsembleAvg,
)


def passage_match(passages: list[str], answers: list[str]) -> int:
    """Returns True if any of the passages contains the answer."""
    has_ans = [dsp.passage_has_answers(psg, answers) for psg in passages]
    if True in has_ans:
        return has_ans.index(True)
    return -1


# def get_data_for_eval():
#     with open("data/hotpotqa_train_1k.jsonl", "r") as f:
#         data = [ujson.loads(line) for line in f.readlines()]
#         train = [(sample["question"], sample["answers"]) for sample in data]
#         dev = [(sample["question"], sample["answers"]) for sample in data]

#     train = [
#         dsp.Example(question=question, answer=answer) for question, answer in train
#     ]
#     dev = [dsp.Example(question=question, answer=answer) for question, answer in dev]
#     return train, dev

def get_data_for_eval():
    with open("data/hotpotqa_train_1k.jsonl", "r") as f:
        data = [ujson.loads(line) for line in f.readlines()]
        data = [(sample["question"], sample["answers"]) for sample in data]

    train = [
        dsp.Example(question=question, answer=answer) for question, answer in data
    ]
    return train


# def evaluateReranker(train, dev, reranker):
#     data = []

#     for example in tqdm.tqdm(dev):
#         question = example.question

#         config = {
#             "org_ranking_fnc": dsp.retrieve,
#             "org_ranking_params": {"k": 100},
#             "new_ranking_fnc": retrieveRerank,
#             "new_ranking_params": {"k": 100, "reranker": reranker},
#         }

#         prediction, prediction_with_reranker = multihop_QA(
#             question, train=train, num_queries=1, num_preds=0, config=config
#         )

#         d = dict(example)

#         prediction_rank_h0 = passage_match(
#             prediction["h0_copy"].context, example.answer
#         )
#         prediction_rank_h1 = passage_match(
#             prediction["h1_copy"].context, example.answer
#         )
#         d["h0_rr"] = 0 if prediction_rank_h0 == -1 else 1 / (prediction_rank_h0 + 1)
#         d["h1_rr"] = 0 if prediction_rank_h1 == -1 else 1 / (prediction_rank_h1 + 1)

#         prediction_reranker_rank_h0 = passage_match(
#             prediction_with_reranker["h0_copy"].context, example.answer
#         )
#         prediction_reranker_rank_h1 = passage_match(
#             prediction_with_reranker["h1_copy"].context, example.answer
#         )
#         d["reranker_h0_rr"] = (
#             0
#             if prediction_reranker_rank_h0 == -1
#             else (1 / (prediction_reranker_rank_h0 + 1))
#         )
#         d["reranker_h1_rr"] = (
#             0
#             if prediction_reranker_rank_h1 == -1
#             else (1 / (prediction_reranker_rank_h1 + 1))
#         )

#         # d['prediction'] = prediction.answer
#         d["sucess@7"] = dsp.passage_match(prediction.context, example.answer)
#         d["reranker_sucess@7"] = dsp.passage_match(
#             prediction_with_reranker.context, example.answer
#         )
#         data.append(d)

    # df = pd.DataFrame(data)

    # # percentage = round(100.0 * df['correct'].sum() / len(dev), 1)
    # # print(f"Answered {df['correct'].sum()} / {len(dev)} ({percentage}%) correctly.")

    # df["sucess@7"] = df["sucess@7"].apply(lambda x: "✔️" if x else "❌")
    # df["reranker_sucess@7"] = df["reranker_sucess@7"].apply(
    #     lambda x: "✔️" if x else "❌"
    # )

    # print(
    #     "Baseline MRR:\n\nHop 0: ",
    #     np.average(df["h0_rr"]),
    #     "\nHop 1: ",
    #     np.average(df["h1_rr"]),
    #     "\n---\n",
    # )
    # print(
    #     "Reranker MRR:\n\nHop 0: ",
    #     np.average(df["reranker_h0_rr"]),
    #     "\nHop 1: ",
    #     np.average(df["reranker_h1_rr"]),
    #     "\n---\n",
    # )
    # pd.options.display.max_colwidth = None
    # display(
    #     df.style.set_table_styles(
    #         [
    #             {"selector": "th", "props": [("text-align", "left")]},
    #             {"selector": "td", "props": [("text-align", "left")]},
    #         ]
    #     )
    # )
    # return df

def evaluateRerankerBatchedResults(data, print_df=False):
    df = pd.DataFrame(data)

    # percentage = round(100.0 * df['correct'].sum() / len(dev), 1)
    # print(f"Answered {df['correct'].sum()} / {len(dev)} ({percentage}%) correctly.")
    success_7 = df["success@7"].value_counts()[True]
    reranker_success_7 = df["reranker_success@7"].value_counts()[True]
    
    df["success@7"] = df["success@7"].apply(lambda x: "✔️" if x else "❌")
    df["reranker_success@7"] = df["reranker_success@7"].apply(
        lambda x: "✔️" if x else "❌"
    )

    print(
        "Baseline:\n\nHop 0 MRR: ",
        np.average(df["h0_rr"]),
        "\nHop 1 MRR: ",
        np.average(df["h1_rr"]),
        "\nSuccess@7: ",
        success_7,
        "\n---\n",
    )
    print(
        "Reranker:\n\nHop 0 MRR: ",
        np.average(df["reranker_h0_rr"]),
        "\nHop 1 MRR: ",
        np.average(df["reranker_h1_rr"]),
        "\nSuccess@7: ",
        reranker_success_7,
        "\n---\n",
    )
    
    pd.options.display.max_colwidth = None
    if print_df:
        display(
            df.style.set_table_styles(
                [
                    {"selector": "th", "props": [("text-align", "left")]},
                    {"selector": "td", "props": [("text-align", "left")]},
                ]
            )
        )
    return df

def evaluateRerankerBatched(train, dev, reranker):
    data = []

    for example in tqdm.tqdm(dev):
        question = example.question

        config = {
            "org_ranking_fnc": dsp.retrieve,
            "org_ranking_params": {"k": 100},
            "new_ranking_fnc": retrieveRerank,
            "new_ranking_params": {"k": 100, "reranker": reranker},
        }

        prediction = multihop_QA(
            question, train=train, num_queries=1, num_preds=0, config=config
        )

        d = dict(example)

        prediction_rank_h0 = passage_match(
            prediction["h0_copy_org"].context, example.answer
        )
        prediction_rank_h1 = passage_match(
            prediction["h1_copy_org"].context, example.answer
        )
        d["h0_rr"] = 0 if prediction_rank_h0 == -1 else 1 / (prediction_rank_h0 + 1)
        d["h1_rr"] = 0 if prediction_rank_h1 == -1 else 1 / (prediction_rank_h1 + 1)

        prediction_reranker_rank_h0 = passage_match(
            prediction["h0_copy_reranked"].context, example.answer
        )
        prediction_reranker_rank_h1 = passage_match(
            prediction["h1_copy_reranked"].context, example.answer
        )
        d["reranker_h0_rr"] = (
            0
            if prediction_reranker_rank_h0 == -1
            else (1 / (prediction_reranker_rank_h0 + 1))
        )
        d["reranker_h1_rr"] = (
            0
            if prediction_reranker_rank_h1 == -1
            else (1 / (prediction_reranker_rank_h1 + 1))
        )

        # d['prediction'] = prediction.answer
        d["success@7"] = dsp.passage_match(prediction["h1_copy_org"].context[:7], example.answer)
        d["reranker_success@7"] = dsp.passage_match(
            prediction["h1_copy_reranked"].context[:7], example.answer
        )
        data.append(d)
    return data


# def evaluate_default(max_train_size=10, max_dev_size=10, gpt_model="text-davinci-002", reranker=None):
def evaluate_default(dev_samples_size=10, batch_size_for_eval=5, gpt_model="text-davinci-002", reranker=None):
    # train, dev = get_data_for_eval()
    reranker = load_models(gpt_model=gpt_model, reranker=reranker)
    
    data = get_data_for_eval()
    results = []
    progress = tqdm.tqdm()
    dev_samples_size = min(dev_samples_size, len(data))
    progress.total = dev_samples_size
    for batch_start in range(0, dev_samples_size, batch_size_for_eval):
        batch_end = batch_start + batch_size_for_eval
        samples = data[batch_start:batch_end]
        np.random.shuffle(samples)
        
        train_sample_percentage = 0.1
        while True:
            train_size = int(len(samples) * train_sample_percentage)
            if train_size >= 7:
                break
            train_sample_percentage += 0.1
            if train_sample_percentage >= 0.9:
                raise Exception("Not enough samples for training")
        
        train_samples = samples[:train_size]
        dev_samples = samples[train_size:]
        
        print(f"Data Samples size:\n> Train: {len(train_samples)}\n> dev: {len(dev_samples)}")

        results.extend(evaluateRerankerBatched(train_samples, dev_samples, reranker))
        pickle.dump(results, open("reranker_results.pkl", "wb"))
        progress.update(batch_size_for_eval)
        print(f"Report at {int(batch_end/dev_samples_size)}:")
        _ = evaluateRerankerBatchedResults(results)
        
    
    df = evaluateRerankerBatchedResults(results, print_df=True)
    return df


def load_models(gpt_model="text-davinci-002", reranker=None):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.api_base = os.environ["OPENAI_API_BASE"]
    colbert_server = (
        "http://ec2-44-228-128-229.us-west-2.compute.amazonaws.com:8893/api/search"
    )

    if gpt_model == "gpt-3.5-turbo":
        lm = dsp.GPT3(model=gpt_model, model_type="chat")
    else:
        lm = dsp.GPT3(model=gpt_model)
    rm = dsp.ColBERTv2(url=colbert_server)

    dsp.settings.configure(lm=lm, rm=rm)
    if reranker is None:
        reranker = SentenceTransformersCrossEncoder()
    else:
        reranker = reranker()
    return reranker
