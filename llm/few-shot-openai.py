import openai

import pandas as pd
import numpy as np
import random
import math

import os
import argparse, yaml

from openai import OpenAI

def eval_comp(df_val):
    df = df_val[(df_val["completion_pred"]!="N/A")]

    df["precision"] = None
    df["recall"] = None

    for idx in range(len(df)):
        completion = list(df["completion"])[idx].replace("[", "").replace("]", "").replace("'", "").split(",")
        completion = [mat.strip() for mat in completion]
        completion_pred = list(df["completion_pred"])[idx].replace("[", "").replace("]", "").replace("'", "").split(",")
        completion_pred = [mat.strip() for mat in completion_pred]

        count_shared = len(set(completion).intersection(completion_pred))
        df["precision"][idx] = count_shared/len(completion_pred)
        df["recall"][idx]    = count_shared/len(completion)

    precision = df["precision"].mean()
    recall = df["recall"].mean()
    F1 = 2*precision*recall / (precision+recall)

    accuracy = {"precision": precision,
                "recall": recall,
                "F1": F1}
    print(accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="~/multitask/with_unknown/configs/pythia-2.8b-3task-MC.yaml")
    args = parser.parse_args()

    with open(args.config_path) as cf_file:
        config = yaml.safe_load(cf_file.read())
        data_path = config["dir"]["data"]
        key_path = config["dir"]["key"]
        save_dir = config["dir"]["save"]
        scheme = config["scheme"]["name"]
        shot = config["scheme"]["shot"]

    df_val = pd.read_csv(data_path)

    key = open(key_path, "r").read().strip("\n")
    client = OpenAI(api_key=key)

    df_val["completion_pred"] = "N/A"

    for idx in range(len(df_val)):
        if shot == 0:
            prompt = df_val["prompt"][idx]
        else:
            prompt = df_val["prompt_new"][idx]
        
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        )
        
        df_val["completion_pred"][idx] = response.choices[0].message.content
    

    name = scheme + "_{}".format(shot) + "shot"
    save_path = save_dir + "{}/".format(name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df_val.to_csv(save_path + "df_val.csv")

    eval_comp(df_val)