import pandas as pd
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import random
from tqdm.notebook import tqdm
import evaluate
from peft import PeftModel, PeftModelForCausalLM, PeftConfig

import argparse, yaml
import os


if __name__ == '__main__':
    # 1. load data
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="~/multitask/with_unknown/configs/pythia-2.8b-3task-MC.yaml")
    args = parser.parse_args()

    with open(args.config_path) as cf_file:
        config = yaml.safe_load(cf_file.read())
        task = config["task"]["name"]
        model_path= config["model"]["path"]
        scheme = config["scheme"]["name"]  # ItemInstruction or MultipleChoice
        if_unknown = config["scheme"]["if_unknown"]  # with_unknown or without_unknown
        cache_dir = config["dir"]["cache_dir"]
        data_super_dir = config["dir"]["data_super_dir"]
        model_super_dir = config["dir"]["model_super_dir"]
        num_epochs = config["hyperparams"]["num_train_epochs"]
        lr = config["hyperparams"]["learning_rate"]
        weight_decay = config["hyperparams"]["weight_decay"]
        batch_size = config["hyperparams"]["batch_size"]

    model_name = model_path.split("/")[-1]
    model_dir = model_super_dir + "{}/{}/{}-{}-SFT/".format(if_unknown, scheme, model_name, task)
    data_dir = data_super_dir + "{}/data/{}/".format(if_unknown, scheme)

    df_train = pd.read_csv(data_dir + "df_train.csv")
    df_val = pd.read_csv(data_dir + "df_val.csv")


    save_steps = 4*int(len(df_train)/batch_size)  # save checkpoint every 4 epochs
    final_step = num_epochs*int(len(df_train)/batch_size)
    checkpoint_list = [steps for steps in range(final_step, 0, -save_steps)]

    if scheme == "MultipleChoice":
        train_ids = random.sample(range(len(df_train)), 500)
        val_ids = random.sample(range(len(df_val)), 500)
    else:
        train_ids = [i for i in range(len(df_train))]
        val_ids = [i for i in range(len(df_val))]

    # 2. load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    # 3. generate completions for examples
    if not os.path.exists(model_dir + "Completions"):
        os.makedirs(model_dir + "Completions")

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
    
    
