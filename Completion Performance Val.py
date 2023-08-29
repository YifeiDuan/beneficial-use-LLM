import pandas as pd
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import random
from tqdm.notebook import tqdm

model_name = "pythia-2.8b"
#checkpoint_list = ["500", '2000', "4000"]
checkpoint_list = ["500", '2000', "4000", "1000", '1500', "2500", "3000", "3500"]

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
    val_idx = random.sample(range(0, 15), 5)
    # val_idx = [10, 11, 4, 14, 8]
    print(val_idx)
    val_samples = []
    for idx in val_idx:
        val_samples.append(
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
        for i in tqdm(range(len(val_samples)), total=len(val_samples)):
            input_ids = tokenizer.encode(val_samples[i]["prompt"], return_tensors='pt').to("cuda")
            beam_output = model.generate(
                            input_ids,
                            max_length=1024,
                            num_beams=5, 
                            early_stopping=True
                            )

            comp = tokenizer.decode(beam_output[0], skip_special_tokens=True)

            val_samples[i]["completion_pred"] = comp
        
        print("Start writing txt file: ")
        f= open(model_dir + "/Sample Completions/{}_block_{}_val.txt".format(model_name, checkpoint),"w+")
        i = 1
        for paper in tqdm(val_samples, total=len(val_samples)):
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

