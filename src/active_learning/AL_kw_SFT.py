import transformers
import math
from datasets import load_dataset, Dataset
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

def kw_acc_calc(df):
    df["precision"] = None
    df["recall"] = None

    for idx in range(len(df)):
        comp_true = list(df["comp_true"])[idx].replace("[", "").replace("]", "").replace("'", "").split(",")
        comp_true = [mat.strip() for mat in comp_true]
        comp_pred = list(df["comp_pred"])[idx].replace("[", "").replace("]", "").replace("'", "").split(",")
        comp_pred = [mat.strip() for mat in comp_pred]
        
        count_shared = len(set(comp_true).intersection(comp_pred))
        df["precision"][idx] = count_shared/len(comp_true)
        df["recall"][idx]    = count_shared/len(comp_pred)

        kw_acc = {}
        for kw in ["k1", "k2", "k3", "k4", "k5"]:
            kw_precision = df[(df["kw_group"]==kw)]["precision"].mean()
            kw_recall = df[(df["kw_group"]==kw)]["recall"].mean()
            kw_F1 = 2*kw_precision*kw_recall/(kw_precision+kw_recall)
            kw_acc[kw] = kw_F1

    return kw_acc


def kw_sample(df_train_prev, df_val_fix, df_pool, kw_acc, step=10):
    n_new = step
    df_val = df_val_fix
    sorted_kw = [val[0] for val in sorted(kw_acc.items(), key = lambda x: (x[1], x[0]))]
    df_train_var = []
    df_val_var = []
    for kw in sorted_kw:
        df_pool_kw = df_pool[(df_pool["kw_group"]==kw)]
        if n_new == 0:
            df_val_var.append(df_pool_kw)
        else:
            if len(df_pool_kw) <= n_new:
                df_train_var.append(df_pool_kw)
                n_new -= len(df_pool_kw)
            else:
                df_train_var.append(df_pool_kw.loc[:(n_new-1)])
                df_val_var.append(df_pool_kw.loc[n_new:])
                n_new = 0
        
    df_train = pd.concat([df_train_prev] + df_train_var, ignore_index=True)
    df_val = pd.concat([df_val_fix] + df_val_var, ignore_index=True)
    df_newPool = pd.concat(df_val_var, ignore_index=True)

    return df_train, df_val, df_newPool


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="~/multitask/ActiveLearning/kw_baseline/configs/AL_config.yaml")
    args = parser.parse_args()

    with open(args.config_path) as cf_file:
        config = yaml.safe_load(cf_file.read())
        train_size  = config['AL_hyper']['train_size']
        step = config['AL_hyper']['step']
        task = config["task"]["name"]
        model_checkpoint = config["model"]["path"]
        cache_dir = config["dir"]["cache_dir"]
        data_dir = config["dir"]["data_dir"]
        model_dir = config["dir"]["model_dir"]
        num_epochs = config["hyperparams"]["num_train_epochs"]
        lr = config["hyperparams"]["learning_rate"]
        weight_decay = config["hyperparams"]["weight_decay"]
        batch_size = config["hyperparams"]["batch_size"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, cache_dir=cache_dir)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
    )

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, cache_dir=cache_dir)
    model = get_peft_model(model, peft_config).to("cuda")

    model_name = model_checkpoint.split("/")[-1]

    prev_train_size = train_size - step
    prev_save_steps = 4*math.ceil(prev_train_size/batch_size)
    prev_checkpoint = 9*prev_save_steps
    comp_val_prev = pd.read_csv(model_dir + "{}-{}-SFT-{}/".format(model_name, task, prev_train_size) + "Completions/{}_val.csv".format(prev_checkpoint))

    # calculate F1 of each kw group
    kw_acc = kw_acc_calc(comp_val_prev)

    # read in previous train and val sets
    df_train_prev = pd.read_csv(data_dir+"df_AL_kw_train_{}.csv".format(prev_train_size))
    df_val_fix = pd.read_csv(data_dir+"df_AL_val_fix.csv")
    df_pool = pd.read_csv(data_dir+"df_AL_kw_PoolFor_{}.csv".format(train_size))
   

    # sample based on kw heuristics
    df_train, df_val, df_newPool = kw_sample(df_train_prev, df_val_fix, df_pool, kw_acc)
    df_train.to_csv(data_dir + "df_AL_kw_train_{}.csv".format(train_size), index=False)
    df_val.to_csv(data_dir + "df_AL_kw_val_{}.csv".format(train_size), index=False)
    df_newPool.to_csv(data_dir + "df_AL_kw_PoolFor_{}.csv".format(train_size+step), index=False)
    
    datasets = load_dataset("csv", data_files={"train": data_dir+"df_AL_kw_train_{}.csv".format(train_size), "val": data_dir+"df_AL_kw_val_{}.csv".format(train_size)})


    response_template = " ### Completion:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


    save_steps = 4*math.ceil(len(df_train)/batch_size)

    training_args = TrainingArguments(
        f"{model_dir}{model_name}-{task}-SFT-{train_size}",
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