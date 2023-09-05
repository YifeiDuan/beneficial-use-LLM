import transformers
from datasets import load_dataset
from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML
from transformers import Trainer, TrainingArguments

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

cache_dir = "/dccstor/yifei01/.cache/huggingface/"
model_checkpoint = "EleutherAI/pythia-2.8b"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, cache_dir=cache_dir)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
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


def formatting_prompts_func(examples):
    output_texts = []
    for i in range(len(examples['prompt'])):
        text = f"### Prompt: {examples['prompt'][i]}\n ### Completion: {examples['completion'][i]}"
        output_texts.append(text)
    return output_texts


if __name__ == '__main__':
    print("transformers version: ".format(transformers.__version__))


    datasets = load_dataset("csv", data_files={"train": "df_sum_train.csv", "val": "df_sum_val.csv"})

    #show_random_elements(datasets["train"])

    response_template = " ### Completion:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    model_name = model_checkpoint.split("/")[-1]

    #logger = HFSummaryWriter(repo_id=f"/dccstor/yifei01/ bu-{model_name}-block/log", commit_every=30)

    training_args = TrainingArguments(
        f"/dccstor/yifei01/bu-{model_name}-single",
        evaluation_strategy = "epoch",
        num_train_epochs=100,
        learning_rate=2e-5,
        weight_decay=0.01,
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