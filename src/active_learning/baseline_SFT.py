import transformers
from datasets import load_dataset, Dataset
from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML
from transformers import Trainer, TrainingArguments

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import torch
from peft import get_peft_config, get_peft_model, PromptTuningInit, LoraConfig, TaskType, PeftType

import argparse, yaml

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def formatting_prompts_func(examples):
    output_texts = []
    for i in range(len(examples['prompt'])):
        text = f"### Instructions: {examples['prompt'][i]}\n ### Completion: {examples['completion'][i]}"
        output_texts.append(text)
    return output_texts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="~/multitask/ActiveLearning/configs/AL_config.yaml")
    args = parser.parse_args()

    with open(args.config_path) as cf_file:
        config = yaml.safe_load(cf_file.read())
        train_size  = config['hyper']['train_size']
        cache_dir = config['dir']['cache_dir']
        data_dir = config['dir']['data_dir']
        model_checkpoint = config['model']['path']
        task = config['task']['name']
        num_train_epochs = config['hyper']['num_train_epochs']
        batch_size = config['hyper']['batch_size']
        learning_rate = config['hyper']['learning_rate']
        weight_decay = config['hyper']['weight_decay']
        evaluation_strategy = config['log']['evaluation_strategy']
        logging_strategy = config['log']['logging_strategy']

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, cache_dir=cache_dir)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
    )

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, cache_dir=cache_dir)
    model = get_peft_model(model, peft_config).to(DEVICE)

    df_train = pd.read_csv(data_dir+"df_train_AL.csv")

    df_train_AL = df_train[df_train.index.isin([i for i in range(train_size)])]
    df_train_AL.to_csv(data_dir + "df_train_AL_{}.csv".format(train_size), index=False)
    
    datasets = load_dataset("csv", data_files={"train": data_dir+"df_train_AL_{}.csv".format(train_size), "val": data_dir+"df_val_AL.csv"})


    response_template = " ### Completion:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    model_name = model_checkpoint.split("/")[-1]

    training_args = TrainingArguments(
        f"/dccstor/yifei01/bu_multitask/ActiveLearning/{model_name}-{task}-SFT-{train_size}",
        evaluation_strategy = evaluation_strategy,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        logging_strategy=logging_strategy,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        save_steps = train_size,
        push_to_hub=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    trainer.train() 