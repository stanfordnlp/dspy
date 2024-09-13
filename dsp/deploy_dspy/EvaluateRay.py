import dspy
from dspy.evaluate import Evaluate
import ray
import threading
import types
import pandas as pd
import math
import tqdm
from IPython.display import HTML, display as ipython_display

# class DSPyActor:
#     def __init__(self, async_mode=False, batch_size=5):
#         dotenv.load_dotenv()
#         print("loading model")
#         model_path = download_model("meta-llama/Meta-Llama-3-8B-Instruct")
#         self.llm = AsyncLLMWrapper(model_path,
#                        max_pending_requests=512,
#                        tokenizer=AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct"),
#                        enforce_eager=True,
#                        engine_use_ray=False,
#                        worker_use_ray=False,
#                        enable_prefix_caching=True,
#                     #    tensor_parallel_size=8
#                     )
#         self.lm = dspy.VLLMOfflineEngine(llm=self.llm, batch_size=batch_size, async_mode=async_mode)
#         COLBERT_V2_ENDPOINT = "http://20.102.90.50:2017/wiki17_abstracts"
#         self.rm = dspy.ColBERTv2(url=COLBERT_V2_ENDPOINT)
#         # self.lm = dspy.MultiOpenAI(model="meta-llama/Meta-Llama-3.1-8B-Instruct", api_key=token, api_base=api_base, api_provider="anyscale")


#         dspy.settings.configure(lm=self.lm, rm=self.rm, experimental=True)

#         self.basic_pred = BasicMH()
#         self.thread_pool = ThreadPoolExecutor(max_workers=20)


#     def __call__(self, inputs):
#         results = {"results": []}
#         futures: list[Future] = []

#         for idx, question in enumerate(inputs["item"]):
#             future = self.thread_pool.submit(self.process_question, question, idx)
#             futures.append(future)
        
#         for future in futures:
#             result = future.result()
#             results["results"].append(result)

#         return results

#     def process_question(self, question, idx):
#         # time.sleep(10 * idx)
#         with dspy.context(lm=self.lm):
#             try:
#                 pred = self.basic_pred(question=question)
#                 return {"question": question, "answer": pred.answer}
#             except Exception as e:
#                 print("error", traceback.print_exception(e))
#                 return {"question": question, "error": str(e)}

# def get_results(dataset, start_idx, end_idx):
#     ds = ray.data.from_items([x.question for x in dataset[start_idx:end_idx]])
#     concurrency = 8
#     batch_size = math.ceil(ds.count() / concurrency)
#     results = ds.map_batches(DSPyActor,
#             batch_size=batch_size,
#             num_gpus=1,
#             concurrency=concurrency,
#             fn_constructor_kwargs={"batch_size": batch_size}
#         ).take_all()
#     return results

class EvaluateDSPyActor:
    def __init__(self, engine_args):
        self.engine_args = engine_args
    
    def __call__(self, inputs):
        return self.engine_args(inputs)
    

# TODO: Fix caching

"""
What do I care about in order to run evaluate with ray?
1. I need to know which LM to instantiate in each actor (can I get this from settings?)
5. Does the metric need to run on a GPU?

2. How many GPUs each actor needs - x
3. How many threads to use for each actor - x
4. How many actors to create - x

I can get 2,3,4 from the evaluateRay function

bsz should always be 256

Would it ever make sense to make some sort of LM object that you put in settings that isnt really an LM object, and is moreso a primitive that contains information for how to create that LM in the batch offline setting?

Just spitballing here momentarily


For the batch offline inference case, you have a (total GPUs/GPUs per instance) different LLM engine instances that get created, and then you have one DSPy OfflineLM instance per LLM Engine instance. Right now I instantiate the LLM engine and pass it to the OfflineLM. This pattern is relatively hard to avoid, but you could instantiate the engine inside of the OfflineLM. Is there anything stopping you from doing so inside the LM object? You effectively treat the LM and the Engine like singletons anyways. Maybe have the option to do both. 


"""
class EvaluateRay(Evaluate):
    def __init__(self,
        *,
        devset,
        metric=None,
        num_threads=1,
        display_progress=False,
        display_table=False,
        max_errors=5,
        return_all_scores=False,
        return_outputs=False,
        **_kwargs,
        ):
        super().__init__(devset=devset, metric=metric, num_threads=-1, display_progress=display_progress, display_table=display_table, max_errors=max_errors, return_all_scores=return_all_scores, return_outputs=return_outputs, **_kwargs)

    def __call__(
        self,
        program,
        metric=None,
        devset=None,
        num_threads=None,
        display_progress=None,
        display_table=None,
        return_all_scores=None,
        return_outputs=None,
    ):
        metric = metric if metric is not None else self.metric
        devset = devset if devset is not None else self.devset
        num_threads = num_threads if num_threads is not None else self.num_threads
        display_progress = display_progress if display_progress is not None else self.display_progress
        display_table = display_table if display_table is not None else self.display_table
        return_all_scores = return_all_scores if return_all_scores is not None else self.return_all_scores
        return_outputs = return_outputs if return_outputs is not None else self.return_outputs
        results = []

        

        devset = list(enumerate(devset))
        tqdm.tqdm._instances.clear()

        if num_threads == 1:
            reordered_devset, ncorrect, ntotal = self._execute_single_thread(wrapped_program, devset, display_progress)
        else:
            # reordered_devset, ncorrect, ntotal = self._execute_multi_thread(
            #     wrapped_program,
            #     devset,
            #     num_threads,
            #     display_progress,
            # )
            ds = ray.data.from_items([x.question for x in devset])
            concurrency = 8
            batch_size = math.ceil(ds.count() / concurrency)
            results = ds.map_batches(DSPyActor,
                    batch_size=batch_size,
                    num_gpus=1,
                    concurrency=concurrency,
                    fn_constructor_kwargs={"batch_size": batch_size}
                ).take_all()

        dspy.logger.info(f"Average Metric: {ncorrect} / {ntotal} ({round(100 * ncorrect / ntotal, 1)}%)")

        predicted_devset = sorted(reordered_devset)

        if return_outputs:  # Handle the return_outputs logic
            results = [(example, prediction, score) for _, example, prediction, score in predicted_devset]

        data = [
            merge_dicts(example, prediction) | {"correct": score} for _, example, prediction, score in predicted_devset
        ]

        result_df = pd.DataFrame(data)

        # Truncate every cell in the DataFrame (DataFrame.applymap was renamed to DataFrame.map in Pandas 2.1.0)
        result_df = result_df.map(truncate_cell) if hasattr(result_df, "map") else result_df.applymap(truncate_cell)

        # Rename the 'correct' column to the name of the metric object
        metric_name = metric.__name__ if isinstance(metric, types.FunctionType) else metric.__class__.__name__
        result_df = result_df.rename(columns={"correct": metric_name})

        if display_table:
            if isinstance(display_table, bool):
                df_to_display = result_df.copy()
                truncated_rows = 0
            else:
                df_to_display = result_df.head(display_table).copy()
                truncated_rows = len(result_df) - display_table

            styled_df = configure_dataframe_display(df_to_display, metric_name)

            ipython_display(styled_df)

            if truncated_rows > 0:
                # Simplified message about the truncated rows
                message = f"""
                <div style='
                    text-align: center;
                    font-size: 16px;
                    font-weight: bold;
                    color: #555;
                    margin: 10px 0;'>
                    ... {truncated_rows} more rows not displayed ...
                </div>
                """
                ipython_display(HTML(message))

        if return_all_scores and return_outputs:
            return round(100 * ncorrect / ntotal, 2), results, [score for *_, score in predicted_devset]
        if return_all_scores:
            return round(100 * ncorrect / ntotal, 2), [score for *_, score in predicted_devset]
        if return_outputs:
            return round(100 * ncorrect / ntotal, 2), results

        return round(100 * ncorrect / ntotal, 2)