import pandas as pd
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import random
from tqdm.notebook import tqdm
import evaluate
from peft import PeftModel, PeftModelForCausalLM, PeftConfig
import torch

import argparse, yaml
import math
import os


if __name__ == '__main__':
    # 1. configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="~/multitask/with_unknown/configs/pythia-2.8b-3task-MC.yaml")
    args = parser.parse_args()

    with open(args.config_path) as cf_file:
        config = yaml.safe_load(cf_file.read())
        SFT_task = config["task"]["SFT"]
        tokenizer_path = config["model"]["path"]
        version = config["model"]["version"]
        checkpoint = config["model"]["checkpoint"]
        scheme = config["scheme"]["name"]  # ItemInstruction or MultipleChoice
        if_unknown = config["scheme"]["if_unknown"]  # with_unknown or without_unknown
        cache_dir = config["dir"]["cache_dir"]
        analysis_data_dir = config["dir"]["embed_data_dir"]
        model_super_dir = config["dir"]["model_super_dir"]

    model_name = tokenizer_path.split("/")[-1]
    if version == "finetuned":       # fine-tuned model stored in server
        model_path = model_super_dir + f"{if_unknown}/{scheme}/{model_name}-{SFT_task}-SFT/checkpoint-{checkpoint}/"
    elif version == "pretrained":       # pre_trained model, model path and tokenizer path are the same
        model_path = tokenizer_path

    save_path = analysis_data_dir + "embeddings/"


    # 2. load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    # 3. prepare data
    df = pd.read_csv(analysis_data_dir + f"df_mat_app_new_dropunknown.csv")
    
    
    # 4. load finetuned model
    config = PeftConfig.from_pretrained(model_path)
    if version == "finetuned":
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path).to("cuda")
        model = PeftModelForCausalLM.from_pretrained(model, model_path).to("cuda")
    elif version == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")


    # 5. perform embeddings and save
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    embedding_records = []

    for i in tqdm(range(len(df))):
        if i%100 == 0:
            print(i)
        row = df.iloc[i]
        text = row["prompt"]
        input_ids = tokenizer(text, return_tensors="pt").to("cuda")
        
        outputs = model(**input_ids, output_hidden_states=True, use_cache=False)
        last_layer_hidden_states = outputs["hidden_states"][-1].contiguous().cpu().detach()
        paragraph_embedding = last_layer_hidden_states[ :, -1, : ].contiguous()
        
        record = {
            "mat": row["mat"],
            "app": row["app"],
            "paragraph_embedding": paragraph_embedding
        }
        
        embedding_records.append(record)

    embeddings = torch.cat([item["paragraph_embedding"] for item in embedding_records])
    torch.save(embeddings, save_path + f"{model}_{version}_checkpoint{checkpoint}.pt")
    
    
