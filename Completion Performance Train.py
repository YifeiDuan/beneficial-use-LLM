import pandas as pd
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import random
from tqdm.notebook import tqdm

model_name = "pythia-2.8b"
#checkpoint_list = ["500", '2000', "4000"]
checkpoint_list = ["1000", '1500', "2500", "3000", "3500"]

data_dir = "/u/duanyf99/pythia/"
tokenizer_dir = "EleutherAI/"
model_dir = "/dccstor/yifei01/bu-{}-block/".format(model_name)

if __name__ == '__main__':
    # 1. load data
    df_train = pd.read_csv(data_dir + "df_sum_train.csv")
    df_val = pd.read_csv(data_dir + "df_sum_val.csv")

    # 2. load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir + model_name, use_fast=True)

    # 3. sample training examples
    #train_idx = random.sample(range(0, 85), 10)
    train_idx = [26, 73, 31, 66, 84, 78, 53, 33, 57, 30]
    print(train_idx)
    train_samples = []
    for idx in train_idx:
        train_samples.append(
            {
                "prompt": list(df_train["prompt"])[idx] + "\n" + "Completion: [{'material': {'mat_name': ",
                "completion_true": list(df_train["completion"])[idx]
            }
        )

    # 4. (for every selected checkpoint) infer completions from tuned model
    for checkpoint in checkpoint_list:
        print("Checkpoint: {}".format(checkpoint))

        model = AutoModelForCausalLM.from_pretrained(model_dir + 'checkpoint-{}'.format(checkpoint)).to("cuda")
        print("Start inferring for training examples: ")
        for i in tqdm(range(len(train_samples)), total=len(train_samples)):
            input_ids = tokenizer.encode(train_samples[i]["prompt"], return_tensors='pt').to("cuda")
            beam_output = model.generate(
                            input_ids,
                            max_length=1024,
                            num_beams=5, 
                            early_stopping=True
                            )

            comp = tokenizer.decode(beam_output[0], skip_special_tokens=True)

            train_samples[i]["completion_pred"] = comp
        
        print("Start writing txt file: ")
        f= open(model_dir + "/Sample Completions/{}_block_{}_train.txt".format(model_name, checkpoint),"w+")
        i = 1
        for paper in tqdm(train_samples, total=len(train_samples)):
            f.write("Example %d\r\n" % (i))
            f.write("True:\n")
            f.write(paper["prompt"] + paper["completion_true"].replace("[{'material': {'mat_name': ", ""))
            f.write("\n")
            f.write("\n")
            f.write("Pred:\n")
            f.write(paper["completion_pred"])
            f.write("\n")
            f.write("\n")
            f.write("\n")

            i = i+1

