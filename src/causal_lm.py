import transformers
from datasets import load_dataset
from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
import torch

from transformers.integrations import TensorBoardCallback

import argparse, yaml

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
block_size = 128

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))


def tokenize_function(examples):
    return tokenizer(examples["text"])

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="./configs/CLM.yaml")

    args = parser.parse_args()

    with open(args.config_path) as cf_file:
        config = yaml.safe_load(cf_file.read())
        cache_dir = config['dir']['cache_dir']
        data_dir = config['dir']['data_dir']
        model_checkpoint = config['model']['path']
        proc_batch_size = config['hyper']['proc_batch_size']
        num_proc = config['hyper']['num_proc']
        num_train_epochs = config['hyper']['num_train_epochs']
        learning_rate = config['hyper']['learning_rate']
        weight_decay = config['hyper']['weight_decay']
        evaluation_strategy = config['log']['evaluation_strategy']
    

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, cache_dir=cache_dir).to(DEVICE)

    print("transformers version: ".format(transformers.__version__))


    datasets = load_dataset("csv", data_files={"train": data_dir + "df_gen_train.csv", "validate": data_dir + "df_gen_val.csv"})

    #show_random_elements(datasets["train"])

    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["DOI", "text"])
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=proc_batch_size,
        num_proc=num_proc,
    )

    model_name = model_checkpoint.split("/")[-1]

    #logger = HFSummaryWriter(repo_id=f"/dccstor/yifei01/ bu-{model_name}-block/log", commit_every=30)

    training_args = TrainingArguments(
        f"/dccstor/yifei01/bu-{model_name}-block",
        evaluation_strategy = evaluation_strategy,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validate"],
        callbacks = [TensorBoardCallback]
    )

    trainer.train()