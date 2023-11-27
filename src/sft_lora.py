import transformers
from datasets import load_dataset
from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML
from transformers import Trainer, TrainingArguments

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from peft import get_peft_config, get_peft_model, PromptTuningInit, LoraConfig, TaskType, PeftType

import argparse, yaml


def formatting_prompts_func(examples):
    output_texts = []
    for i in range(len(examples['prompt'])):
        text = f"### Instructions: {examples['prompt'][i]}\n ### Completion: {examples['completion'][i]}"
        output_texts.append(text)
    return output_texts


if __name__ == '__main__':
    print("transformers version: ".format(transformers.__version__))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="~/multitask/with_unknown/configs/pythia-2.8b-3task-MC.yaml")
    args = parser.parse_args()

    with open(args.config_path) as cf_file:
        config = yaml.safe_load(cf_file.read())
        task = config["task"]["name"]
        model_checkpoint = config["model"]["path"]
        scheme = config["scheme"]["name"]  # ItemInstruction or MultipleChoice
        if_unknown = config["scheme"]["if_unknown"]  # with_unknown or without_unknown
        cache_dir = config["dir"]["cache_dir"]
        data_super_dir = config["dir"]["data_super_dir"]
        model_super_dir = config["dir"]["model_super_dir"]
        num_epochs = config["hyperparams"]["num_train_epochs"]
        lr = config["hyperparams"]["learning_rate"]
        weight_decay = config["hyperparams"]["weight_decay"]
        batch_size = config["hyperparams"]["batch_size"]
        save_epoch_step = config["save"]["every_num_epoch"]

    model_dir = model_super_dir + "{}/{}/".format(if_unknown, scheme)
    data_dir = data_super_dir + "{}/data/{}/".format(if_unknown, scheme)


    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
    )

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, cache_dir=cache_dir)
    model = get_peft_model(model, peft_config).to("cuda")

    df_train = pd.read_csv(data_dir+"df_train.csv")
    save_steps = save_epoch_step*int(len(df_train)/batch_size)


    datasets = load_dataset("csv", data_files={"train": data_dir+"df_train.csv", "val": data_dir+"df_val.csv"})


    response_template = " ### Completion:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    model_name = model_checkpoint.split("/")[-1]

    training_args = TrainingArguments(
        f"{model_dir}/{model_name}-{task}-SFT",
        evaluation_strategy = "epoch",
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        logging_strategy="epoch",
        save_steps = save_steps,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
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