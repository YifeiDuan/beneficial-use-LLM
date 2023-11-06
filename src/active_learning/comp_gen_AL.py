import pandas as pd
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import random
from tqdm.notebook import tqdm
import evaluate
from peft import PeftModel, PeftModelForCausalLM, PeftConfig

import argparse, yaml
import os


task = "3task"
cache_dir = "/dccstor/yifei01/.cache/huggingface/"
data_dir = "/u/duanyf99/multitask/ActiveLearning/Data/"

model_name = "pythia-2.8b"

tokenizer_dir = "EleutherAI/"


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="~/multitask/ActiveLearning/configs/AL_config.yaml")
    args = parser.parse_args()

    with open(args.config_path) as cf_file:
        config = yaml.safe_load(cf_file.read())
        train_size  = config['hyper']['train_size']
    

    model_dir = "/dccstor/yifei01/bu_multitask/ActiveLearning/{}-{}-SFT-{}/".format(model_name, task, train_size)
    if not os.path.exists(model_dir + "Completions"):
        os.makedirs(model_dir + "Completions")
    checkpoint_list = [steps for steps in range(9*train_size, 0, -train_size)]

    # 1. load data
    df_train = pd.read_csv(data_dir + "df_train_AL_{}.csv".format(train_size))
    df_val = pd.read_csv(data_dir + "df_val_AL.csv")

    train_ids = random.sample(range(len(df_train)), 100)
    val_ids = random.sample(range(len(df_val)), 100)

    # 2. load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir + model_name, use_fast=True, cache_dir=cache_dir)

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
    for checkpoint in checkpoint_list:
        print("Checkpoint: {}".format(checkpoint))

        config = PeftConfig.from_pretrained(model_dir + 'checkpoint-{}/'.format(checkpoint))
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path).to("cuda")
        model = PeftModelForCausalLM.from_pretrained(model, 
                                                    model_dir + 'checkpoint-{}/'.format(checkpoint)).to("cuda")

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
        df_train_comp.to_csv(model_dir + "Completions/{}_train.csv".format(checkpoint), index=False)

    ## 4.2 val
        print("Checkpoint: {}".format(checkpoint))

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
        df_val_comp.to_csv(model_dir + "Completions/{}_val.csv".format(checkpoint), index=False)
    
    
