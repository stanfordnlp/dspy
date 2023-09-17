import openai
import time

def finetune_open_ai(training_data_path, target):
    file = openai.File.create(
        file=open(training_data_path, "rb"),
        purpose='fine-tune'
    )
    file_id = file.id
    while file.status != "processed":
        time.sleep(5)
        file = openai.File.retrieve(file.id)
    
    response = openai.FineTuningJob.create(training_file=file_id, model=target)
    job_id = response["id"]    
    retrieve_response = openai.FineTuningJob.retrieve(job_id)

    while retrieve_response.status != "succeeded":
        time.sleep(5) 
        retrieve_response = openai.FineTuningJob.retrieve(job_id)

    fine_tuned_model_id = retrieve_response["fine_tuned_model"]
    return fine_tuned_model_id
