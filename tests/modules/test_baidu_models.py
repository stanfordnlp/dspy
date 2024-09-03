"""Tests for baidu ernie models.
Note: Requires configuration of your baidu qianfan app api_key and secret_key.
https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application/v1
"""

import dspy

models = {
    "ERNIE-4.0-8K": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions",
    "ERNIE-4.0-Turbo-8K": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-4.0-turbo-8k"
}


def get_lm(name: str) -> dspy.LM:
    return dspy.Baidu(model=name)


def run_tests():
    """Test the providers and models"""
    models

    predict_func = dspy.Predict("question -> answer")
    for model_name in models.keys():
        lm = get_lm(model_name)
        with dspy.context(lm=lm):
            question = "文心千帆属于哪个公司？"
            answer = predict_func(question=question).answer
            print(f"Question: {question}\nAnswer: {answer}")
            print("---------------------------------")
            lm.inspect_history()
            print("---------------------------------\n")


if __name__ == "__main__":
    run_tests()
