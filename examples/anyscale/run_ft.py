from dspy import TrainableAnyscale
from dsp.modules.lm import TrainingMethod
from vllm import SamplingParams
import msgspec
import ujson
import anyscale
from anyscale.job.models import JobConfig, JobState
import os
import jsonlines
# from dspy.



def main():
    train_path = "train_data.json"
    eval_path = "val_data.json"
    method = TrainingMethod.SFT
    lm = TrainableAnyscale(model="meta-llama/Meta-Llama-3-8B-Instruct")

    # lm.kwargs["model"] = "meta-llama/Meta-Llama-3-8B-Instruct"

    train_dataset = lm._format_data_for_vanilla_finetuning(train_path)
    val_dataset = lm._format_data_for_vanilla_finetuning(eval_path) if eval_path else None

    if not lm._verify_datasets(train_dataset, val_dataset):
        print("Unable to verify arguments")
        raise RuntimeError("Unable to verify argument")

    formatted_paths = {}
    for path, dataset in [(train_path, train_dataset), (eval_path, val_dataset)]:
        if not (path and dataset):
            continue
        formatted_path = path.split(".")[0] + "_formatted.jsonl"
        with open(formatted_path, "w") as f:
            for item in dataset:
                f.write(ujson.dumps(item) + "\n")

        with jsonlines.open(formatted_path) as reader:
            print("num items in ", path, ": ", sum(1 for _ in reader))
        
        formatted_paths[path] = formatted_path

    s3_train_path, s3_eval_path = lm._submit_data(train_path=formatted_paths[train_path], eval_path=formatted_paths[eval_path])
    
    kwargs = {
        "hyperparameters": {
            "num_devices": 4,
            "trainer_resources": None,
            "worker_resources": None
        }
    }
    compute_config_path, compute_config = lm._generate_config_files(use_lora=True, train_path=s3_train_path, eval_path=s3_eval_path, **kwargs)

    if method != TrainingMethod.SFT:
        raise NotImplementedError("Only SFT finetuning is supported at the moment.")
    # job_id: str = anyscale.job.submit(
    #     compute_config
    # )
    # anyscale.job.wait(id=job_id, timeout_s=18000)
    # print(f"Job {job_id} succeeded!")
    command = compute_config.entrypoint
    print(command)
    os.system(command)

    # model_info = anyscale.llm.models.get(job_id=job_id).to_dict()
    # print(model_info)




        
if __name__ == "__main__":
    main()