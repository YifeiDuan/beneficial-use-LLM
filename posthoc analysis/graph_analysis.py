import numpy as np
import pandas as pd
import math
import os
import networkx as nx
import matplotlib.pyplot as plt


def jaccard_node_similarity():
    # Load graph
    save_dir = "Saved Data/G_withDOI_v0"
    G = nx.read_graphml(save_dir + "Graph.graphml")

    # Load data
    df = pd.read_csv("../Data (Reformatted)/df_all.csv")
    all_mat = set(df[(df["mat"]!="unknown")]["mat"].dropna())
    all_app = set(df[(df["app"]!="unknown")]["app"].dropna())
    all_prod = set(df[(df["prod"]!="unknown")]["prod"].dropna())


    # Material Nodes
    mat_nodes = [node for node, data in G.nodes(data=True) if data.get("node_type") == "material"]
    mat_pairs = []
    for mat1 in mat_nodes:
        for mat2 in mat_nodes:
            mat_pairs.append((mat1, mat2))


    # Compute Jaccard similarity
    mat_similarity_jaccard = nx.jaccard_coefficient(G, mat_pairs)
    mat_jaccard_records = []
    for u, v, p in mat_similarity_jaccard:
        record = {
            "material1": u,
            "material2": v,
            "jaccard":   p
        }
        mat_jaccard_records.append(record)

    df_mat_jaccard = pd.DataFrame.from_records(mat_jaccard_records)

    # Save data
    df_mat_jaccard.to_csv(save_dir + "MAT_Jaccard.csv", index=False)

    
    
    
    
    
    # Plot
    df_mat_jaccard = pd.read_csv(save_dir + "MAT_Jaccard.csv")

    data = df_mat_jaccard.pivot(index="material1", columns="material2", values="jaccard")


    # Set mask
    lower_triangle_indices = np.tril_indices(data.shape[0])
    mask = np.ones(data.shape, dtype=bool)
    mask[lower_triangle_indices] = False


    import seaborn as sns
    sns.set(rc={'figure.figsize':(15, 12)})
    heatmap = sns.heatmap(data, vmin=0, vmax=1, 
                        cmap="hot",
                        cbar_kws={'shrink': 0.3,
                                    'label': "Jaccard Coefficient (Node Similarity)"}, 
                        mask=mask,
                        xticklabels=True, yticklabels=True)

    fig = heatmap.get_figure()

    import os
    fig_dir = "../Figs/Graph & Link Pred/"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    fig.savefig(fig_dir + "mat_jaccard_heatmap_lower_triangle.jpg", bbox_inches="tight")