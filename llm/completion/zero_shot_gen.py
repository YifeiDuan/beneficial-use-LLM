import pandas as pd
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import random
from tqdm.notebook import tqdm
import evaluate
from peft import PeftModel, PeftModelForCausalLM, PeftConfig
import torch

import argparse, yaml

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="./configs/comp_gen/task_gen.yaml")

    args = parser.parse_args()

    with open(args.config_path) as cf_file:
        config = yaml.safe_load(cf_file.read())
        cache_dir = config['dir']['cache_dir']
        data_dir = config['dir']['data_dir']
        tokenizer_dir = config['dir']['tokenizer_dir']
        model_sup_dir = config['dir']['model_sup_dir']
        model_name = config['model']['name']
        task = config['task']['name']
    
    model_dir = model_sup_dir + "/{}-{}-SFT/".format(model_name, task)

    # 1. load data
    df_train = pd.read_csv(data_dir + "df_train.csv")
    df_val = pd.read_csv(data_dir + "df_val.csv")

    train_ids = random.sample(range(len(df_train)), 100)
    val_ids = random.sample(range(len(df_val)), 100)

    # 2. load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir + model_name, use_fast=True)

    # 3. generate completions for examples
    # 3.1 train
    train_samples = []
    for idx in train_ids:
        train_samples.append(
            {
                "DOI": list(df_train["DOI"])[idx],
                "prompt": "### Instructions: " + list(df_train["prompt"])[idx] + "\n" + "### Completion: ",
                "comp_true": list(df_train["completion"])[idx].strip()
            }
        )
    # 3.2 val
    val_samples = []
    for idx in val_ids:
        val_samples.append(
            {
                "DOI": list(df_val["DOI"])[idx],
                "prompt": "### Instructions: " + list(df_val["prompt"])[idx] + "\n" + "### Completion: ",
                "comp_true": list(df_val["completion"])[idx].strip()
            }
        )
    
    # 4. (for every selected checkpoint) infer completions from tuned model
    ## 4.1 train
    model = AutoModelForCausalLM.from_pretrained(model_dir + model_name, cache_dir=cache_dir).to("cuda")

    print("Start inferring for training examples: ")
    for i in tqdm(range(len(train_samples)), total=len(train_samples)):
        input_ids = tokenizer.encode(train_samples[i]["prompt"], return_tensors='pt').to("cuda")
        beam_output = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=128,
                        num_beams=5, 
                        early_stopping=True
                        )

        comp = tokenizer.decode(beam_output[0], skip_special_tokens=True)

        comp = comp.split("### Completion: ")[1].split("]")[0] + "]"
        comp = comp.strip()

        train_samples[i]["comp_pred"] = comp

    df_train_comp = pd.DataFrame.from_records(train_samples)
    df_train_comp.to_csv(model_dir+ "Completions/0_train.csv", index=False)

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
    df_val_comp.to_csv(model_dir + "Completions/0_val.csv", index=False)

    
