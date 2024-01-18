import transformers
import math
from datasets import load_dataset, Dataset
from datasets import ClassLabel
import random
from tqdm.notebook import tqdm
import pandas as pd
from IPython.display import display, HTML
from transformers import Trainer, TrainingArguments

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from peft import get_peft_config, get_peft_model, PromptTuningInit, LoraConfig, TaskType, PeftType, PeftModelForCausalLM, PeftConfig

import argparse, yaml
import os

import torch.nn as nn
import itertools
import math

from utils import conditional_perplexity

def formatting_prompts_func(examples):
    output_texts = []
    for i in range(len(examples['prompt'])):
        text = f"### Instructions: {examples['prompt'][i]}\n ### Completion: {examples['completion'][i]}"
        output_texts.append(text)
    return output_texts



def condPP_calc(df, model, tokenizer, semantic_class):
    df["condPP"] = None
    for idx in range(len(df)):
        comp_pred = list(df["comp_pred"])[idx].replace("[", "").replace("]", "").replace("'", "").split(",")
        comp_pred = [mat.strip() for mat in comp_pred]
        context = list(df["prompt"])[idx]

        condPP = conditional_perplexity(model, tokenizer, context, comp_pred, semantic_class=semantic_class)
        df["condPP"][idx] = condPP
    
    return df


def condPP_sample(df_train_prev, df_val_prev_with_condPP, step=10):

    df_val_prev_with_condPP = df_val_prev_with_condPP.sort_values(by=["condPP"], ascending=False)
    df_val_prev_with_condPP = df_val_prev_with_condPP.reset_index().drop(columns=["comp_pred"]).rename(columns={"comp_true":"completion"})
    df_train_add = df_val_prev_with_condPP.loc[:(step-1)]
        
    df_train = pd.concat([df_train_prev, df_train_add], ignore_index=True)
    df_val = df_val_prev_with_condPP.loc[step:]

    return df_train, df_val




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
        semantic_class = config["hyperparams"]["semantic_class"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, cache_dir=cache_dir)

    if semantic_class is True:
        PP_strategy = "semantic"
    else:
        PP_strategy = "list"

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

    # calculate conditional perplexity of the val pool (previous val set)
    df_val_prev_with_condPP = condPP_calc(comp_val_prev, model, tokenizer, semantic_class)

    # read in previous train sets
    df_train_prev = pd.read_csv(data_dir+"df_AL_condPP_train_{}.csv".format(prev_train_size))
    #df_val_prev = pd.read_csv(data_dir+"df_AL_condPP_val_{}.csv".format(prev_train_size))
   

    # sample based on kw heuristics
    df_train, df_val = condPP_sample(df_train_prev, df_val_prev_with_condPP)
    df_train.to_csv(data_dir + "df_AL_condPP_train_{}.csv".format(train_size), index=False)
    df_val.to_csv(data_dir + "df_AL_condPP_val_{}.csv".format(train_size), index=False)
    
    datasets = load_dataset("csv", data_files={"train": data_dir+"df_AL_condPP_train_{}.csv".format(train_size), "val": data_dir+"df_AL_condPP_val_{}.csv".format(train_size)})


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








    # generate completions
    model_cp_dir = model_dir + "{}-{}-SFT-{}/".format(model_name, task, train_size)

    if not os.path.exists(model_cp_dir + "Completions"):
        os.makedirs(model_cp_dir + "Completions")
    
    #save_step = train_size
    checkpoint_list = [steps for steps in range(9*save_steps, 6*save_steps, -save_steps)]

    # 1. load data

    train_ids = random.sample(range(len(df_train)), 100)
    val_ids = [i for i in range(len(df_val))]

    # 2. load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, cache_dir=cache_dir)

    # 3. generate completions for examples
    
    # 3.2 val
    val_samples = []
    for idx in val_ids:
        val_samples.append(
            {
                "DOI": list(df_val["DOI"])[idx],
                "prompt": "### Instructions: " + list(df_val["prompt"])[idx] + "\n" + "### Completion: ",
                "kw_group": list(df_val["kw_group"])[idx],
                "comp_true": list(df_val["completion"])[idx].strip()
            }
        )
    
    # 4. (for every selected checkpoint) infer completions from tuned model
    ## 4.1 train
    for checkpoint in checkpoint_list:
        print("Checkpoint: {}".format(checkpoint))

        config = PeftConfig.from_pretrained(model_cp_dir + 'checkpoint-{}/'.format(checkpoint))
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path).to("cuda")
        model = PeftModelForCausalLM.from_pretrained(model, 
                                                    model_cp_dir + 'checkpoint-{}/'.format(checkpoint)).to("cuda")

    ## 4.2 val

        print("Start inferring for valing examples: ")
        for i in tqdm(range(len(val_samples)), total=len(val_samples)):
            input_ids = tokenizer.encode(val_samples[i]["prompt"], return_tensors='pt').to("cuda")
            beam_output = model.generate(
                            input_ids=input_ids,
                            max_new_tokens=128,
                            num_beams=5, 
                            early_stopping=True
                            )

            comp = tokenizer.decode(beam_output[0], skip_special_tokens=True)

            comp = comp.split("### Completion: ")[1].split("]")[0] + "]"
            comp = comp.strip()

            val_samples[i]["comp_pred"] = comp
            
        df_val_comp = pd.DataFrame.from_records(val_samples)
        df_val_comp.to_csv(model_cp_dir + "Completions/{}_val.csv".format(checkpoint), index=False)