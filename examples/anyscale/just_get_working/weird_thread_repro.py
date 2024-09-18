from dsp.deploy_dspy.async_llm import AsyncLLMWrapper
import dspy

model = "meta-llama/Meta-Llama-3-8B-Instruct"
engine_args = {
    "max_pending_requests": 512,
    "enforce_eager": True,
    "engine_use_ray": False,
    "worker_use_ray": False,
    "enable_prefix_caching": True,
    "tensor_parallel_size": 1,
    "enable_lora": True
}

lm = dspy.VLLMOfflineEngine.instantiate_with_llm(model=model, engine_args=engine_args)
dspy.settings.configure(lm=lm)
predictor = dspy.Predict("question -> answer")

print(predictor(question="What is the capital of France?"))

# lm.shutdown()