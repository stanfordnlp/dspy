# import ray
# from fastapi import FastAPI
# from ray import serve

# import dspy
# import dsp
# from concurrent.futures import Future
# import time
# from typing import Any, List, Optional, Literal, Union
# import ujson
# import openai
# from collections import defaultdict
# import yaml # note the extra import
# import dotenv
# from pydantic import BaseModel
# import concurrent.futures
# import traceback


# from vllm import LLM


# import os



# ray.init(runtime_env={"py_modules": [dspy, dsp]})

# class DSPyActor:
#     def __init__(self):
#         dotenv.load_dotenv()
#         api_base = os.environ.get("API_BASE")
#         token = os.environ.get("OPENAI_API_KEY")
#         self.llm = LLM("meta-llama/Meta-Llama-3-8B-Instruct",
#                        enforce_eager=True,
#         #   enable_prefix_caching=True,
#         )
#         self.lm = dspy.VLLMOfflineEngine(llm=self.llm)

#         # self.lm = dspy.MultiOpenAI(model="meta-llama/Meta-Llama-3.1-8B-Instruct", api_key=token, api_base=api_base, api_provider="anyscale")


#         dspy.settings.configure(lm=self.lm)

#         self.basic_pred = dspy.ChainOfThought("question -> answer")

#     def __call__(self, question):
#         with dspy.context(lm=self.lm):
#             try:
#                 pred = self.basic_pred(question=question)
#             except Exception as e:
#                 print(traceback.print_exception(e))
#                 # print("Error in prediction:", e.with_traceback(), question)
#                 return {"error": str(e)}
#             return pred
    
# app = FastAPI()
# class QuestionRequest(BaseModel):
#     question: str

# @serve.deployment(route_prefix="/", num_replicas=1, ray_actor_options={"num_cpus": 6, "num_gpus": 1})
# @serve.ingress(app)
# class DSPyServedDeployment:
#     def __init__(self):
#         self.actor = DSPyActor()
        
#     @app.post("/predict")
#     def predict(self, request: QuestionRequest):
#         return self.actor(request.question)


# def send_request(example):
#     try:
#         response = requests.post("http://127.0.0.1:8000/predict", json={"question": example.question})
#         response.raise_for_status()  # Raise an error for bad HTTP status codes
#         return response.json(), example
#     except requests.RequestException as e:
#         print(f"Request failed for example {example}: {e}")
#         return None, example
    
# if __name__ == "__main__":
#     deployment = DSPyServedDeployment.bind()
#     serve.run(deployment, blocking=False)

#     import requests
#     import time
#     num = 1
#     num_requests = 10
#     dataset: list[dspy.Example] = [dspy.Example(question=f"What is {num} + {x}?" , answer=f"{num+x}").with_inputs("question") for x in range(num_requests)]
#     question_answer_map = {example.question: example for example in dataset}
#     start_time = time.time()
#     total = 0
#     metric = dspy.evaluate.answer_exact_match

#     # Use ThreadPoolExecutor to send requests concurrently
#     with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
#         # Start the load operations and mark each future with its example
#         futures = [executor.submit(send_request, example.inputs()) for example in dataset]
        
#         # wait for all futures to complete
#         futures = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED).done

#         for future in futures:
#             response_json, example = future.result()
            
#             if response_json is not None:
#                 try:
#                     pred = dspy.Example(**example, **response_json)
#                     # actual = question_answer_map[example.question]
#                     # correct = metric(actual, pred)
#                     # total += correct
#                 except Exception as e:
#                     print(f"Error processing example {example}: {e}, response: {response_json}")

#     end_time = time.time()

#     print(f"Total correct: {total}")
#     print(f"Time taken: {end_time - start_time} seconds")
#     print(f"Time per request: {(end_time - start_time) / num_requests} seconds")

#     # response = requests.post("http://127.0.0.1:8000/predict", json=data)
#     # print(response.json())
#     serve.shutdown()
#     print("Serve deployment has been shut down.")

# """
# 1 replica, local VLLM, 1 concurrent requests: 4.56s per request
# 1 replica, playground model, 10 concurrent requests: 0.31s per request
# 2 replicas, playground model, 10 concurrent requests: 0.13s per request
# """