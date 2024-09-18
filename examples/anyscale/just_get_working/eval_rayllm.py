# from openai import OpenAI
import dspy

# TODO: Replace this model ID with your own.
# MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct:isaac:twzsb"

def eval_rayllm():
    normal_llama = "meta-llama/Meta-Llama-3-70B-Instruct"
    lm = dspy.MultiOpenAI(
        model=normal_llama,
        api_base="https://endpoints-v2-enough-gray-iron-mglb8.cld-tffbxe9ia5phqr1u.s.anyscaleuserdata.com/v1",
        api_key="sk-anyscale-api-key",
        api_provider="vllm",
        model_type="chat",
    )
    dspy.settings.configure(lm=lm)

    predict = dspy.Predict("question -> answer")

    print(predict(question="what is the capital of France?").answer)

if __name__ == "__main__":
    eval_rayllm()
    # query("http://localhost:8000", "sk-anyscale-api-key")