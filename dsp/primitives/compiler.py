import os
import time
import tqdm
import ujson
import random
import subprocess
import os
from dsp.utils.settings import CompilerConfig

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

import dsp
from datasets.fingerprint import Hasher

if os.environ.get('DSP_NOTEBOOK_CACHEDIR'):
    training_data_directory = os.path.join(os.environ.get('DSP_NOTEBOOK_CACHEDIR'), 'compiler')
else:
    training_data_directory = 'cache/compiler'


compilations_assumed_to_exist={'ft-zvEdzQVQ5xwlxvNPrxl6kpnw': 'ada:ft-stanfordpraglab-2023-02-09-19-50-49'}


def openai_check_finetune(jobname):
    if dsp.settings.force_reuse_cached_compilation and jobname in compilations_assumed_to_exist:
        return compilations_assumed_to_exist[jobname]

    command = f"""openai api fine_tunes.get -i {jobname}"""
    print(command)

    result = subprocess.run(command.split(), stdout=subprocess.PIPE)
    output = result.stdout.decode("utf-8").strip()

    try:
        output = ujson.loads(output)
        if output['status'] == 'succeeded':
            return output['fine_tuned_model']

        if output['status'] in ['pending', 'running']:
            print(f'Compiling, run ```openai api fine_tunes.follow -i {jobname}``` for details...')
            time.sleep(60)
            return openai_check_finetune(jobname)
    except:
        pass

    return False


def convert_to_training_point2(y, inputs, outputs, template):
    assert len(inputs) + len(outputs) == len(template.fields)

    y_ = dsp.Example(**{f: y[f] for f in inputs}, demos=[])
    prompt = template(y_, show_guidelines=False)

    completion = y[outputs[0]]
    output_fields = template.fields[len(inputs):]

    for field in output_fields[1:]:
        completion += f"\n\n{field.name} " + y[field.output_variable]
    
    completion = " " + completion + " </s>"
    return {'prompt': prompt, 'completion': completion}


def simulate(program, input_examples):
    training_data = []

    for input_example in tqdm.tqdm(input_examples):
        prediction = program(input_example)

        if prediction is not None:
            # assert len(prediction.compiling_stages) == 2, "TMP"
            for stage in prediction.compiling_stages:
                name, template, inputs, outputs = stage['name'], stage['template'], stage['inputs'], stage['outputs']
                training_data.append(convert_to_training_point2(prediction.get(name), inputs, outputs, template))
    
    r = random.Random(0)
    r.shuffle(training_data)

    return training_data


def openai_finetune_(name, target):
    training_data_path = name_to_path(name)

    # Launch the fine-tune on the path
    command = f"""openai api fine_tunes.create -t {training_data_path} -m {target} --n_epochs 4 --learning_rate_multiplier 0.05 --no_check_if_files_exist"""
    print(command)

    # command = """python script.py"""
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    while line := process.stdout.readline().decode().strip():
        if 'created fine-tune:' in line.lower():
            jobname = line.split()[-1]
            break
        
    #     if 'costs $' in line.lower():
    #         cost = line.split()[-1]
    #         break

    # assert cost[0] == '$'
    
    # if float(cost[1:]) > 300:
    #     print(f'Got cost {cost} -- you may wanna cancel the job: openai api fine_tunes.cancel -i {jobname}')

    # print(cost)

    print(jobname)

    # Block until it's done
    ft = openai_check_finetune(jobname)
    assert ft, ft

    # Return its name
    return (jobname, ft)


def openai_finetune(name, target):
    print(name)
    training_data_path = name_to_path(name)
    training_data_path += '.model'

    # if path + stuff exists, load the tuple from it
    try:
        with open(training_data_path) as f:
            jobname, ft = ujson.loads(f.readline())

        if openai_check_finetune(jobname):
            return jobname, ft
    except:
        pass
    
    jobname, ft = openai_finetune_(name, target)

    with open(training_data_path, 'w') as f:
        f.write(ujson.dumps((jobname, ft)) + '\n')
    
    return jobname, ft


def name_to_path(name):
    if not os.path.exists(training_data_directory):
        os.makedirs(training_data_directory)

    training_data_path = os.path.join(training_data_directory, f'{name}.jsonl')
    return training_data_path


# 3. Check that the output file name has status "success" (not deleted or non-existent). Otherwise, re-call with n = n+1.
def finetune(training_data, target):
    name = Hasher.hash(training_data)
    training_data_path = name_to_path(name)

    with open(training_data_path, 'w') as f:
        for line in training_data:
            f.write(ujson.dumps(line) + '\n')

    if provider == 'openai':
        jobname, ft = openai_finetune(name, target)
    
        print(ft)

        ft = dsp.GPT3(model=ft, stop=" </s>")
    elif provider == 'hf':
        config = dsp.settings.compiler_config
        # TODO: The new model should be loaded with the hf model loading wrapper
        ft_path = hf_finetune(training_data, config)
        ft = dsp.HFModel(ft_path)
    return ft

# 4. Return updated program.
def compile(program, examples, target='ada', provider='openai'):
    training_data = simulate(program, examples)
    compiled_lm = finetune(training_data, target=target, provider=provider)

    def compiled_program(*args, **kwargs):
        with dsp.settings.context(compiled_lm=compiled_lm, compiling=False):
            return program(*args, **kwargs)

    compiled_program.lm = compiled_lm
    return compiled_program

def update_lora_config(config: CompilerConfig, train_dataset: Dataset):
    # choose the lora_alpha based on the size of the dataset
    len_dataset = len(train_dataset)
    if len_dataset < 10:
        config["lora_alpha"] = 32
        config["lora_r"] = 16
    elif len_dataset < 100:
        config["lora_alpha"] = 64
        config["lora_r"] = 32
    elif len_dataset < 1000:
        config["lora_alpha"] = 256
        config["lora_r"] = 128
    else:
        config["lora_alpha"] = 512
        config["lora_r"] = 256
    return config

def hf_finetune(train_dataset, config: CompilerConfig):
    
    train_dataset = Dataset.from_list(train_dataset)
    train_dataset = train_dataset.map(lambda examples: {'text': [prompt + response for prompt, response in zip(examples['prompt'], examples['completion'])]}, batched=True)
    
    if config["auto_optimize_lora_params"]:
        config = update_lora_config(config, train_dataset)
    
    compute_dtype = getattr(torch, config["bnb_4bit_compute_dtype"])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["use_4bit"],
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config["use_nested_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        device_map=config["device_map"]
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    peft_config = LoraConfig(
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        r=config["lora_r"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        auto_find_batch_size=config["auto_find_batch_size"],
        optim=config["optim"],
        save_steps=config["save_steps"],
        logging_steps=config["logging_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        fp16=config["fp16"],
        bf16=config["bf16"],
        max_grad_norm=config["max_grad_norm"],
        max_steps=config["max_steps"],
        warmup_ratio=config["warmup_ratio"],
        group_by_length=config["group_by_length"],
        lr_scheduler_type=config["lr_scheduler_type"],
        report_to="all",
        evaluation_strategy="steps",
        eval_steps=25
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        tokenizer=tokenizer,
        args=training_arguments,
        packing=config["packing"],
    )
    trainer.train()
    model_save_path = os.path.join(config["output_dir"], config["new_model"])
    trainer.model.save_pretrained(model_save_path)
    return model_save_path