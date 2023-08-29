import transformers
from datasets import load_dataset
from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments

from transformers.integrations import TensorBoardCallback

block_size = 128
cache_dir = "/dccstor/yifei01/.cache/huggingface/"
model_checkpoint = "EleutherAI/pythia-2.8b"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint, cache_dir=cache_dir).to("cuda")

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
    print("transformers version: ".format(transformers.__version__))


    datasets = load_dataset("csv", data_files={"train": "df_gen_train.csv", "validate": "df_gen_val.csv"})

    #show_random_elements(datasets["train"])

    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["DOI", "text"])
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    model_name = model_checkpoint.split("/")[-1]

    #logger = HFSummaryWriter(repo_id=f"/dccstor/yifei01/ bu-{model_name}-block/log", commit_every=30)

    training_args = TrainingArguments(
        f"/dccstor/yifei01/bu-{model_name}-block",
        evaluation_strategy = "epoch",
        num_train_epochs=100,
        learning_rate=2e-5,
        weight_decay=0.01,
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