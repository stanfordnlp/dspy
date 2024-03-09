# Adapted from: https://www.philschmid.de/fine-tune-flan-t5#3-fine-tune-and-evaluate-flan-t5

import copy
import glob
import json
import os
import warnings
from dataclasses import dataclass

import evaluate
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    set_seed,
)

# from peft import get_peft_model, LoraConfig, TaskType
from transformers.trainer_callback import TrainerCallback

# from dsp.modules.finetuning.fid import *


warnings.filterwarnings("ignore")

IGNORE_INDEX = -100
DEFAULT_SEP_TOKEN = "[SEP]"
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "</s>"
SPECIAL_TOKENS_DICT = {
    "sep_token": DEFAULT_SEP_TOKEN,
    "pad_token": DEFAULT_PAD_TOKEN,
    # "eos_token": DEFAULT_EOS_TOKEN,
    # "bos_token": DEFAULT_BOS_TOKEN,
    "unk_token": DEFAULT_UNK_TOKEN,
}


def _freeze_model_layers(model, unfreeze_last_n):
    # Freeze all layers
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Unfreeze the last n transformer blocks in the decoder
    NUM_DECODER_LAYERS = len(model.transformer.h)
    for i, m in enumerate(model.transformer.h):
        if i >= NUM_DECODER_LAYERS - unfreeze_last_n:
            for parameter in m.parameters():
                parameter.requires_grad = True 

    # Unfreeze parameters after decoder block
    for parameter in model.transformer.ln_f.parameters():
        parameter.requires_grad = True
    for parameter in model.lm_head.parameters():        
        parameter.requires_grad = True
    return model


def _load_data(path):
    # dataset = Dataset.from_json(path)
    L = []
    import ujson
    with open(path) as f:
        for line in f:
            L.append(ujson.loads(line))

    dataset = Dataset.from_list(L)
    return dataset


def preprocess_prompt(text, tokenizer, encoder_decoder_model, decoder_only_model, rationale):
    text = f'{text} ' if encoder_decoder_model else f'{text} {tokenizer.sep_token}'
    return text


def preprocess_completion(text, tokenizer, encoder_decoder_model, decoder_only_model, rationale):
    text = text if encoder_decoder_model else f'{text}{tokenizer.eos_token}'
    return text.lstrip()


def _preprocess_data(dataset, tokenizer, encoder_decoder_model, decoder_only_model, config):
    dataset = dataset.map(lambda x: {
        "prompt": preprocess_prompt(x["prompt"], tokenizer, encoder_decoder_model, decoder_only_model, config['rationale']),
        "completion": preprocess_completion(x["completion"], tokenizer, encoder_decoder_model, decoder_only_model, config['rationale']),
    })
    skipped = [x for x in dataset if x["completion"] is None]
    print(f'# examples skipped due to parsing error: {len(skipped)} / {len(dataset)}')
    dataset = dataset.filter(lambda x: x["completion"])
    return dataset


def _tokenize_dataset(dataset, tokenizer, encoder_decoder_model, decoder_only_model):
    def get_dataset_stats(dataset, tokenizer, column):
        tokenized_inputs = dataset.map(lambda x: tokenizer(x[column]), batched=True)
        max_length = max([len(x) for x in tokenized_inputs["input_ids"]])
        return max_length

    def get_tokens_seq2seq(sample, max_source_length, max_target_length, padding="max_length"):
        # Tokenize inputs
        model_inputs = tokenizer(sample["prompt"], max_length=max_source_length, padding=padding, truncation=True)

        # Tokenize targets
        labels = tokenizer(text_target=sample["completion"], max_length=max_target_length, padding=padding, truncation=True)
        labels = labels["input_ids"]

        # Replace all tokenizer.pad_token_id in the labels by IGNORE_INDEX when we want to ignore padding in the loss.
        if padding == "max_length":
            labels = [[(l if l != tokenizer.pad_token_id else IGNORE_INDEX) for l in label] for label in labels]

        model_inputs["labels"] = labels
        return model_inputs

    def get_tokens_causal(sample, max_length, padding="max_length"):
        # Tokenize inputs
        model_inputs = tokenizer(sample["combined"], max_length=max_length, padding=padding, truncation=True)

        # Create targets
        labels = copy.deepcopy(model_inputs["input_ids"])
        prompts = tokenizer(sample["prompt"], max_length=max_length, truncation=True)
        prompt_lens = [len(tokens) for tokens in prompts["input_ids"]]
        for label, source_len in zip(labels, prompt_lens):
            label[:source_len] = [IGNORE_INDEX] * source_len

        # Replace all tokenizer.pad_token_id in the labels by IGNORE_INDEX when we want to ignore padding in the loss.
        if padding == "max_length":
            labels = [[(l if l != tokenizer.pad_token_id else IGNORE_INDEX) for l in label] for label in labels]

        model_inputs["labels"] = labels
        return model_inputs

    if encoder_decoder_model:
        max_source_length = get_dataset_stats(dataset, tokenizer, "prompt")
        max_target_length = get_dataset_stats(dataset, tokenizer, "completion")
        kwargs = {"max_source_length" : max_source_length, "max_target_length" : max_target_length}
        tokenized_dataset = dataset.map(get_tokens_seq2seq, batched=True, fn_kwargs=kwargs)

    elif decoder_only_model:
        dataset = dataset.map(lambda example: {"combined": example["prompt"] + " " + example["completion"]})
        dataset = dataset.filter(lambda x: len(tokenizer(x["combined"])["input_ids"]) <= tokenizer.model_max_length)
        max_length = get_dataset_stats(dataset, tokenizer, "combined")
        kwargs = {"max_length" : max_length}
        tokenized_dataset = dataset.map(get_tokens_causal, batched=True, fn_kwargs=kwargs)

    print(f"Dataset statistics: {kwargs}")
    print(f"Keys of tokenized dataset: {list(tokenized_dataset.features)}")
    return tokenized_dataset


def _compute_metrics(metric, eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace IGNORE_INDEX in the labels as we can't decode them.
    labels = np.where(labels != IGNORE_INDEX, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result


class PeftSavingCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = state.best_model_checkpoint
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(state.best_model_checkpoint, "pytorch_model.bin")
        os.remove(pytorch_model_path) if os.path.exists(pytorch_model_path) else None


def _train_seq2seq(model, tokenizer, tokenized_dataset, metric, config):
    # Define data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['output_dir'],
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        per_device_eval_batch_size=config['batch_size'],
        predict_with_generate=True,
        learning_rate=config['lr'], #1e-4, # 5e-5
        num_train_epochs=config['epochs'],
        # logging & evaluation strategies
        log_level="error",
        logging_dir=f"{config['output_dir']}/logs",
        logging_strategy="steps",
        logging_steps=500,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=config['epochs'],
        load_best_model_at_end=True,
        report_to="tensorboard",
        fp16=config['fp16'],
        bf16=config['bf16'],
    )

    # Create trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        compute_metrics=lambda x: _compute_metrics(metric, x, tokenizer),
        callbacks=[PeftSavingCallback] if config['peft'] else None,
    )

    trainer.train()

    return trainer.state.best_model_checkpoint


def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    """
    Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


@dataclass
class DataCollatorForSupervisedDataset:
    """
    Collate examples for supervised fine-tuning.
    """
    tokenizer: PreTrainedTokenizer

    def __call__(self, instances):
        pad_token_id = self.tokenizer.pad_token_id

        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)

        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(pad_token_id))


def _train_causal(model, tokenizer, tokenized_dataset, metric, config):
    # Define data collator
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # Define training args
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        per_device_eval_batch_size=config['batch_size'],
        learning_rate=config['lr'], #1e-4,# 5e-5
        num_train_epochs=config['epochs'],
        # logging & evaluation strategies
        log_level="error",
        logging_dir=f"{config['output_dir']}/logs",
        logging_strategy="steps",
        logging_steps=500,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=config['epochs'],
        load_best_model_at_end=True,
        report_to="tensorboard",
        fp16=config['fp16'],
        bf16=config['bf16'],
    )

    # Create trainer instance
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        callbacks=[PeftSavingCallback] if config['peft'] else None,
    )

    trainer.train()

    return trainer.state.best_model_checkpoint


def finetune_hf(data_path, target, config):
    set_seed(42)

    output_dir = os.path.join('../finetuning_ckpts', config['save'])

    if os.path.exists(output_dir):
        # training completed, load best model
        ckpts = glob.glob(f'{output_dir}/checkpoint*')
        final_ckpt = sorted(ckpts, key=lambda x: int(x.split('-')[-1]))[-1]
        with open(os.path.join(final_ckpt, 'trainer_state.json')) as f:
            state = json.load(f)
        best_model_checkpoint = state['best_model_checkpoint']

    else:
        os.makedirs(output_dir, exist_ok=True)
        config['target'] = target
        config['output_dir'] = output_dir
        with open(os.path.join(config['output_dir'], 'compiler_config.json'), 'w') as f:
            json.dump(config, f)

        architecture = AutoConfig.from_pretrained(target).__dict__["architectures"][0]
        encoder_decoder_model = ("ConditionalGeneration" in architecture) or ("T5WithLMHeadModel" in architecture)
        decoder_only_model = ("CausalLM" in architecture) or ("GPT2LMHeadModel" in architecture)
        assert encoder_decoder_model or decoder_only_model, f"Unknown HuggingFace model class: {target}"
        assert not config['fid'] or encoder_decoder_model, "Model must be encoder-decoder for Fusion in Decoder"
        assert not config['fid'] or not config['peft'], "FiD and PEFT can't be trained together"

        # load model
        AutoModelClass = AutoModelForSeq2SeqLM if encoder_decoder_model else AutoModelForCausalLM
        if config['peft']:
            model = AutoModelClass.from_pretrained(target, device_map='auto')
            task_type = TaskType.SEQ_2_SEQ_LM if encoder_decoder_model else TaskType.CAUSAL_LM
            peft_config = LoraConfig(task_type=task_type, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        else:
            if config['fid']:
                t5 = AutoModelClass.from_pretrained(target)
                model = FiDT5(t5.config)
                model.load_t5(t5.state_dict())
            else:
                model = AutoModelClass.from_pretrained(target)
                # model = _freeze_model_layers(model, unfreeze_last_n=2)

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(target)
        if decoder_only_model:
            smart_tokenizer_and_embedding_resize(SPECIAL_TOKENS_DICT, tokenizer, model)

        # load data
        dataset = _load_data(data_path)
        dataset = _preprocess_data(dataset, tokenizer, encoder_decoder_model, decoder_only_model, config)
        tokenized_dataset = _tokenize_dataset(dataset, tokenizer, encoder_decoder_model, decoder_only_model)
        tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
        print(f'Finetuning dataset: {tokenized_dataset}')

        # start training
        metric = evaluate.load("rouge")
        if encoder_decoder_model:
            best_model_checkpoint = _train_seq2seq(model, tokenizer, tokenized_dataset, metric, config)
        elif decoder_only_model:
            best_model_checkpoint = _train_causal(model, tokenizer, tokenized_dataset, metric, config)

    print(f'Best checkpoint of model: {best_model_checkpoint}')
    return best_model_checkpoint
