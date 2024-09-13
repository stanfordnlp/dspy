from dspy import TrainableAnyscale
from dsp.modules.lm import TrainingMethod
from vllm import SamplingParams
import msgspec
import ujson
import anyscale
from anyscale.job.models import JobConfig, JobState
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

    for path, dataset in [(train_path, train_dataset), (eval_path, val_dataset)]:
        if not (path and dataset):
            continue
        formatted_path = path.split(".")[0] + "_formatted.jsonl"
        with open(formatted_path, "w") as f:
            for item in dataset:
                f.write(ujson.dumps(item) + "\n")  

    s3_train_path, s3_eval_path = lm._submit_data(train_path=formatted_path, eval_path=formatted_path)

    kwargs = {
        "hyperparameters": {
            "trainer_resources": {
                "memory": 53687091200,
            },
            "worker_resources": {
                "memory": 53687091200,
                "accelerator_type": {
                    "L4": 0.001
                }
            },
        }
    }
    compute_config_path, compute_config = lm._generate_config_files(use_lora=True, train_path=s3_train_path, eval_path=s3_eval_path, **kwargs)

    if method != TrainingMethod.SFT:
        raise NotImplementedError("Only SFT finetuning is supported at the moment.")
    # exit()
    job_id: str = anyscale.job.submit(
        compute_config
    )
    anyscale.job.wait(id=job_id, timeout_s=18000)
    print(f"Job {job_id} succeeded!")

    model_info = anyscale.llm.models.get(job_id=job_id).to_dict()
    print(model_info)




        
if __name__ == "__main__":
    main()