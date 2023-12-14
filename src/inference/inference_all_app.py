import pandas as pd
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import random
from tqdm.notebook import tqdm
import evaluate
from peft import PeftModel, PeftModelForCausalLM, PeftConfig

import argparse, yaml
import math
import os


if __name__ == '__main__':
    # 1. load data
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="~/multitask/with_unknown/configs/pythia-2.8b-3task-MC.yaml")
    args = parser.parse_args()

    with open(args.config_path) as cf_file:
        config = yaml.safe_load(cf_file.read())
        SFT_task = config["task"]["SFT"]
        inference_task = config["task"]["inference"]
        MC_shuffle = config["MC_init"]["shuffle"]
        tokenizer_path = config["model"]["path"]
        checkpoint = config["model"]["checkpoint"]
        scheme = config["scheme"]["name"]  # ItemInstruction or MultipleChoice
        if_unknown = config["scheme"]["if_unknown"]  # with_unknown or without_unknown
        cache_dir = config["dir"]["cache_dir"]
        data_super_dir = config["dir"]["data_super_dir"]
        model_super_dir = config["dir"]["model_super_dir"]

    model_name = tokenizer_path.split("/")[-1]
    model_path = model_super_dir + "{}/{}/{}-{}-SFT/checkpoint-{}/".format(if_unknown, scheme, model_name, SFT_task, checkpoint)
    save_path = model_super_dir + "{}/{}/{}-{}-inference/checkpoint-{}/".format(if_unknown, scheme, model_name, SFT_task, checkpoint)
    data_dir = data_super_dir + "{}/data/{}/Inference/".format(if_unknown, scheme)


    # 2. load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    # 3. generate completions for examples
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 3.1 load data
    df_prompt = pd.read_csv(data_dir + "prompt/{}_prompt_shuffle{}_init{}.csv".format(inference_task, f"{MC_shuffle:02}",f"{MC_shuffle:02}"))
    df_lookup = pd.read_csv(data_dir + "lookup/{}_lookup_shuffle{}_init{}.csv".format(inference_task, f"{MC_shuffle:02}",f"{MC_shuffle:02}"))
    option_to_term = {}
    for i in range(len(df_lookup)):
        term = df_lookup.loc[i, "term"]
        option = df_lookup.loc[i, "option"]
        option = f"{option:02}"
        option_to_term[option] = term
    
    # 3.2 load respective material completion
    df_mat_comp = pd.read_csv(save_path + "mat_comp_shuffle{}.csv".format(f"{MC_shuffle:02}"))


    # 3.3 format data records
    all_info = []
    for idx in range(len(df_prompt)):
        doi = list(df_prompt["DOI"])[idx]
        prompt_template = list(df_prompt["prompt"])[idx]
        mat_comp = list(df_mat_comp[(df_mat_comp["DOI"]==doi)]["comp_term"])[0]
        if mat_comp == "[]":
            continue
        mat_comp = mat_comp.replace("[", "").replace("]", "").replace("'", "").split(",")
        mat_comp = [mat.strip() for mat in mat_comp]

        for mat in mat_comp:
            all_info.append(
                {
                    "DOI": doi,
                    "mat": mat,
                    "prompt": "### Instructions: " + prompt_template.replace("** material **", mat) + "\n" + "### Completion: "
                }
            )
    
    # 4. infer completions from tuned model
    config = PeftConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path).to("cuda")
    model = PeftModelForCausalLM.from_pretrained(model, model_path).to("cuda")

    print("Start inferring for un-annotated papers: ")
    for i in tqdm(range(len(all_info)), total=len(all_info)):
        input_ids = tokenizer.encode(all_info[i]["prompt"], return_tensors='pt').to("cuda")
        beam_output = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=128,
                        num_beams=5, 
                        early_stopping=True
                        )

        comp = tokenizer.decode(beam_output[0], skip_special_tokens=True)

        comp = comp.split("### Completion: ")[1].split("]")[0] + "]"
        comp = comp.strip()

        all_info[i]["comp_option"] = comp
    

    print("Start converting options to terms: ")
    for i in tqdm(range(len(all_info)), total=len(all_info)):
        comp_list_option = all_info[i]["comp_option"].replace("[", "").replace("]", "").replace("\'", "").replace("\"", "").split(",")
        comp_list_option = [option.strip() for option in comp_list_option]
        comp_list_term = [option_to_term[option] for option in comp_list_option if option in option_to_term.keys()]

        all_info[i]["comp_term"] = str(list(set(comp_list_term)))

        
    df_comp = pd.DataFrame.from_records(all_info)
    df_comp.to_csv(save_path + "{}_comp_shuffle{}.csv".format(inference_task, f"{MC_shuffle:02}"), index=False)
    
    
