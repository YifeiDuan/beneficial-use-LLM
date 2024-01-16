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

import torch.nn as nn
import itertools
import math

def conditional_perplexity(model, tokenizer, context, answer, semantic_class=True):
    """
        context: the text before tokenization, not including the answer [...]
        answer: the generated answer [...] as a list
        semantic_class: True or False. if True, the calculated conditional perplexity is for the semantic class, e.g. [1, 2] and [2, 1] are the same semantic class

    """
    tokenized_context = tokenizer(context)
    gen_start_id = len(tokenized_context["input_ids"])
    
    if semantic_class is True:
        semantic_class_prob = 0
        answer_perms = itertools.permutations(answer)
        for answer_item in answer_perms:
            answer_list = list(answer_item)
            answer_text  = f"{answer_list}"
            tokenized_text = tokenizer(context+answer_text, return_tensors="pt")

            logits = model(**tokenized_text)["logits"][0][gen_start_id:]
            softmax = nn.Softmax(dim=1)
            logits = softmax(logits)

            answer_ids = tokenized_text["input_ids"][0][gen_start_id:]
            prob = 1
            
            print(answer_list)
            print(answer_ids)
            for idx in range(len(answer_ids)-1):
                token = answer_ids[idx + 1].item()  ## next token
                logit = logits[idx][token].item()
                prob = prob * logit
            
            semantic_class_prob += prob

        return math.pow(1/semantic_class_prob, 1/len(answer_ids-1))
    else:
        text = context + f"{answer}"
        tokenized_text = tokenizer(text, return_tensors="pt")
        logits = model(**tokenized_text)["logits"][0][gen_start_id:]
        softmax = nn.Softmax(dim=1)
        logits = softmax(logits)

        answer_ids = tokenized_text["input_ids"][0][gen_start_id:]
        prob = 1
        for idx in range(len(answer_ids)-1):
            token = answer_ids[idx + 1]  ## next token
            logit = logits[idx][token]
            prob = prob * logit
        return math.pow(1/prob, 1/len(answer_ids-1))

