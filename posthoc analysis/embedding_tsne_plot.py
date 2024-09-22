import pandas as pd
import numpy as np
import math
import random
import pickle
import torch
import argparse

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



if __name__ == '__main__':
    # 1. configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="e.g. pythia-2.8B, conform to huggingface naming conventions")
    parser.add_argument('--version', type=str, help="pre-trained or fine-tuned")
    parser.add_argument('--checkpoint', type=int, default=7080)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    args = parser.parse_args()

    model = args.model
    version = args.version
    checkpoint = args.checkpoint
    random_state = args.random_state




    # 2. loading both embedding and example text data
    df = pd.read_csv("../Data (Reformatted)/df_mat_app_new_dropunknown.csv")

    embeddings = torch.load(f"embeddings/{model}_{version}_checkpoint{checkpoint}.pt", weights_only=True)



    # 3. embedding dim reduction and matching with labels
    tSNE = TSNE(n_components=2, init='pca', random_state=random_state)
    X_tSNE = tSNE.fit_transform(embeddings)

    df_embed_tSNE = pd.DataFrame({'pc1': X_tSNE[:, 0], 'pc2': X_tSNE[:, 1]})
    df_new = pd.concat([df, df_embed_tSNE], axis=1)


    # 4. example plot for SCM vs geopolymer
    plt.figure(figsize=(10,10))
    for i, APP in enumerate(df_new["app"].dropna().unique()):
        if APP in ["supplementary cementitious material", "geopolymer"]:
            label = "SCM" if APP=="supplementary cementitious material" else APP
                
            df_sub = df_new[(df_new["app"]==APP)]
            plt.scatter(df_sub["pc1"], df_sub["pc2"], alpha=0.5, c=f"C{i}", label=label)

    plt.legend(loc=4, fontsize=22)
    plt.savefig("../Figs/Embeddings/"+ f"{model}_{version}_checkpoing{checkpoint}_tSNE_APP_cem_new.jpg")
    plt.show()