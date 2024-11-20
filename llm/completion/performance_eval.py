import evaluate

import pandas as pd

import argparse, yaml

rouge = evaluate.load("rouge")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="./configs/comp_gen/task_gen.yaml")

    args = parser.parse_args()

    with open(args.config_path) as cf_file:
        config = yaml.safe_load(cf_file.read())
        cache_dir = config['dir']['cache_dir']
        data_dir = config['dir']['data_dir']
        tokenizer_dir = config['dir']['tokenizer_dir']
        model_dir = config['dir']['model_sup_dir']
        model_name = config['model']['name']
        checkpoint_list = config['model']['checkpoint_list']
        task = config['task']['name']


    metrics_train = []
    for checkpoint in checkpoint_list:
        print("checkpoint {}: ".format(checkpoint))

        df_train_comp = pd.read_csv(model_dir + "/Completions/{}_train.csv".format(checkpoint))

        df_train_comp["precision"] = None
        df_train_comp["recall"] = None

        for idx in range(len(df_train_comp)):
            comp_true = list(df_train_comp["comp_true"])[idx].replace("[", "").replace("]", "").replace("'", "").split(",")
            comp_true = [mat.strip() for mat in comp_true]
            comp_pred = list(df_train_comp["comp_pred"])[idx].replace("[", "").replace("]", "").replace("'", "").split(",")
            comp_pred = [mat.strip() for mat in comp_pred]
            
            count_shared = len(set(comp_true).intersection(comp_pred))
            df_train_comp["precision"][idx] = count_shared/len(comp_pred)
            df_train_comp["recall"][idx]    = count_shared/len(comp_true)
        
        # rouge
        rouge_results = rouge.compute(predictions=list(df_train_comp["comp_pred"]), 
                                    references=list(df_train_comp["comp_true"]))
        print(rouge_results)

        # accuracy
        accuracy = {"precision": df_train_comp["precision"].mean(),
                    "recall": df_train_comp["recall"].mean()}
        print(accuracy)
        print("\n")
        
        df_train_comp.to_csv(model_dir + "/Completions/{}_train_metric.csv".format(checkpoint), index=False)

        metrics = dict(rouge_results)
        metrics["checkpoint"] = checkpoint
        metrics["precision"] = df_train_comp["precision"].mean()
        metrics["recall"] = df_train_comp["recall"].mean()
        metrics_train.append(metrics)

    df_metrics_train = pd.DataFrame.from_records(metrics_train)
    df_metrics_train.to_csv(model_dir + "/Completions/metrics_train.csv")




    metrics_val = []
    for checkpoint in checkpoint_list:
        print("checkpoint {}: ".format(checkpoint))
        
        df_val_comp = pd.read_csv(model_dir + "/Completions/{}_val.csv".format(checkpoint))

        df_val_comp["precision"] = None
        df_val_comp["recall"] = None

        for idx in range(len(df_val_comp)):
            comp_true = list(df_val_comp["comp_true"])[idx].replace("[", "").replace("]", "").replace("'", "").split(",")
            comp_true = [mat.strip() for mat in comp_true]
            comp_pred = list(df_val_comp["comp_pred"])[idx].replace("[", "").replace("]", "").replace("'", "").split(",")
            comp_pred = [mat.strip() for mat in comp_pred]

            count_shared = len(set(comp_true).intersection(comp_pred))
            df_val_comp["precision"][idx] = count_shared/len(comp_pred)
            df_val_comp["recall"][idx]    = count_shared/len(comp_true)
        
        # rouge
        rouge_results = rouge.compute(predictions=list(df_val_comp["comp_pred"]), 
                                    references=list(df_val_comp["comp_true"]))
        print(rouge_results)

        # accuracy
        accuracy = {"precision": df_val_comp["precision"].mean(),
                    "recall": df_val_comp["recall"].mean()}
        print(accuracy)
        print("\n")
        
        df_val_comp.to_csv(model_dir + "/Completions/{}_val_metric.csv".format(checkpoint), index=False)

        metrics = dict(rouge_results)
        metrics["checkpoint"] = checkpoint
        metrics["precision"] = df_val_comp["precision"].mean()
        metrics["recall"] = df_val_comp["recall"].mean()
        metrics_val.append(metrics)

    df_metrics_val = pd.DataFrame.from_records(metrics_val)
    df_metrics_val.to_csv(model_dir + "/Completions/metrics_val.csv")
