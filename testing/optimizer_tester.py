from .tasks.scone import ScoNeTask
from .tasks.hotpotqa import HotPotQATask
from .tasks.gsm8k import GSM8KTask
from .tasks.biodex import BioDexTask
from .tasks.tweet import TweetTask
from .tasks.tweet_metric import TweetMetricTask
from dspy.evaluate import Evaluate
from dotenv import load_dotenv
import openai
import os
import dspy
import csv
import datetime
from timeit import default_timer as timer

datasets = ["ScoNe", "HotPotQA", "GSM8K", "BioDex", "Tweet"]

class OptimizerTester:
    def __init__(self, datasets=datasets, default_train_num = 200, default_dev_num = 100, default_test_num = 200,
                 num_threads = 10, default_breadth = 10, default_depth = 3, default_temperature = 1.4,
                 prompt_model_name = "gpt-3.5-turbo-1106", task_model_name = "meta-llama/Llama-2-13b-chat-hf",
                 prompt_model = None, task_model = None,
                 colbert_v2_endpoint = "http://20.102.90.50:2017/wiki17_abstracts"):
        self.datasets = datasets
        self.TRAIN_NUM = default_train_num
        self.DEV_NUM = default_dev_num
        self.EVAL_NUM = default_test_num
        self.NUM_THREADS = num_threads
        self.BREADTH = default_breadth
        self.DEPTH = default_depth
        self.TEMPERATURE = default_temperature
        self.PROMPT_MODEL_NAME = prompt_model_name
        self.TASK_MODEL_NAME = task_model_name
        self.COLBERT_V2_ENDPOINT = colbert_v2_endpoint

        load_dotenv()  # This will load the .env file's variables

        openai.api_key = os.environ.get('OPENAI_API_KEY')
        openai.api_base = os.environ.get('OPENAI_API_BASE')

        # Prompt gen model
        if not prompt_model:
            self.prompt_model = dspy.OpenAI(model=self.PROMPT_MODEL_NAME, max_tokens=150)
        else:
            self.prompt_model = prompt_model

        # Task model
        if not task_model:
            self.task_model = dspy.HFClientTGI(model=self.TASK_MODEL_NAME, port=[7140, 7141, 7142, 7143], max_tokens=150)
        else:
            self.task_model = task_model
        self.colbertv2 = dspy.ColBERTv2(url=colbert_v2_endpoint)

        dspy.settings.configure(rm=self.colbertv2, lm=self.task_model)

    def write_to_csv(self, folder_name, file_name, data):
        # Ensure the output directory exists
        os.makedirs(folder_name, exist_ok=True)
        file_path = os.path.join(folder_name, file_name)
        
        # Check if file exists to determine if headers should be written
        file_exists = os.path.isfile(file_path)

        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            headers = ['test_name', 'train_score', 'dev_score', 'eval_score', 'run_time (sec)', 'train_size', 'dev_size', 'eval_size', 'task_name', 'signature_optimized', 'prompt_model_name', 'task_model_name', 'breadth', 'depth', 'meta_prompt_style', 'fewshot_before', 'fewshot_after', 'temperature', 'fewshot_candidates_num', 'max_bootstrapped_demos', 'bootstrapping', 'view_data', 'optimizer_log_dir', 'additional_notes', 'misc']
            
            # Write headers if the file is being created
            if not file_exists:
                writer.writerow(headers)
            
            # Format the data
            formatted_data = ['NA'] * len(headers)
            formatted_data[-1] = ""
            for key in data.keys():
                if key in headers:
                    formatted_data[headers.index(key)] = data[key]
                else:
                    formatted_data[-1] += f"{key}: {data[key]}\n"
            # Write the data
            writer.writerow(formatted_data)


    def load_dataset(self,dataset):
        ds = None
        dataset = dataset.lower()
        if dataset == "scone":
            ds = ScoNeTask()
        elif dataset == "hotpotqa":
            ds = HotPotQATask()
        elif dataset == "gsm8k":
            ds = GSM8KTask()
        elif dataset == "biodex":
            ds = BioDexTask()
        elif dataset == "tweet":
            ds = TweetTask()
        elif dataset == "tweet_metric":
            ds = TweetMetricTask()
        else:
            raise ValueError("Invalid dataset name.")
        ds.set_splits(TRAIN_NUM=self.TRAIN_NUM, DEV_NUM=self.DEV_NUM, EVAL_NUM=self.EVAL_NUM)
        return ds
        
    # Computes baseline results for a given dataset
    def test_baseline(self, datasets=datasets, test_name="baseline"):
        for dataset in datasets:
            task = self.load_dataset(dataset)
            evaluate_train = Evaluate(devset=task.get_trainset(), metric=task.get_metric(), num_threads=self.NUM_THREADS, display_progress=True)
            evaluate_dev = Evaluate(devset=task.get_devset(), metric=task.get_metric(), num_threads=self.NUM_THREADS, display_progress=True)
            evaluate_eval = Evaluate(devset=task.get_evalset(), metric=task.get_metric(), num_threads=self.NUM_THREADS, display_progress=True)
            default_program = task.get_program()

            # Evaluate the default program
            default_results_train = evaluate_train(default_program)
            default_results_dev = evaluate_dev(default_program)
            default_results_eval = evaluate_eval(default_program)

            # Write the results to a csv
            self.write_to_csv('outputs', 'results.csv', {
                'test_name': dataset + "_" + test_name,
                'train_score': default_results_train,
                'dev_score': default_results_dev,
                'eval_score': default_results_eval,
                'run_time (sec)': 0,
                'train_size': len(task.get_trainset()),
                'dev_size': len(task.get_devset()),
                'eval_size': len(task.get_evalset()),
                'task_name': dataset,
                'signature_optimized': False,
                'prompt_model_name': self.PROMPT_MODEL_NAME,
                'task_model_name': self.TASK_MODEL_NAME,
                'breadth': 'NA',
                'depth': 'NA',
                'meta_prompt_style': 'default',
                'fewshot_before': False,
                'fewshot_after': False,
                'temperature': self.TEMPERATURE,
                'fewshot_candidates_num': 0,
                'max_bootstrapped_demos': 0,
                'bootstrapping': False,
                'view_data': False,
                'optimizer_log_dir': 'NA',
                'additional_notes': '',
                'misc': ''
            })

    def test_optimizer_default(self, optimizer_function, datasets=datasets, test_name="default"):

        for dataset in datasets:
            task = self.load_dataset(dataset)
            evaluate_train = Evaluate(devset=task.get_trainset(), metric=task.get_metric(), num_threads=self.NUM_THREADS, display_progress=True)
            evaluate_dev = Evaluate(devset=task.get_devset(), metric=task.get_metric(), num_threads=self.NUM_THREADS, display_progress=True)
            evaluate_eval = Evaluate(devset=task.get_evalset(), metric=task.get_metric(), num_threads=self.NUM_THREADS, display_progress=True)
            default_program = task.get_program()

            # Set up the optimizer kwargs
            date_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = "log_dir/" + dataset + "_" + test_name + "_" + date_timestamp + "/"
            os.makedirs(log_dir, exist_ok=True)
            kwargs = dict(breadth=self.BREADTH, depth=self.DEPTH, temperature=self.TEMPERATURE, prompt_model=self.prompt_model, view_data=False, log_dir=log_dir, metric=task.get_metric(), task_model=self.task_model)
            
            # Optimize the default program
            start = timer()
            optimized_program, output_dict = optimizer_function(default_program, task.get_trainset(), task.get_devset(), test_name, dataset, kwargs)
            end = timer()

            # Evaluate the optimized program
            optimized_results_train = evaluate_train(optimized_program)
            optimized_results_dev = evaluate_dev(optimized_program)
            optimized_results_eval = evaluate_eval(optimized_program)

            output = {
                'test_name': dataset + "_" + test_name,
                'train_score': optimized_results_train,
                'dev_score': optimized_results_dev,
                'eval_score': optimized_results_eval,
                'run_time (sec)': end - start,
                'train_size': len(task.get_trainset()),
                'dev_size': len(task.get_devset()),
                'eval_size': len(task.get_evalset()),
                'task_name': dataset,
                'signature_optimized': True,
                'prompt_model_name': self.PROMPT_MODEL_NAME,
                'task_model_name': self.TASK_MODEL_NAME,
                'breadth': self.BREADTH,
                'depth': self.DEPTH,
                'meta_prompt_style': 'default',
                'fewshot_before': False,
                'fewshot_after': False,
                'temperature': self.TEMPERATURE,
                'fewshot_candidates_num': 0,
                'max_bootstrapped_demos': 0,
                'bootstrapping': False,
                'view_data': False,
                'optimizer_log_dir': log_dir,
                'additional_notes': '',
                'misc': ''
            }

            output.update(output_dict)

            # Write the results to a csv
            self.write_to_csv('outputs', 'results.csv', output)