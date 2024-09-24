import numpy as np
import pandas as pd
import math
import os
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


def jaccard_node_similarity(save_dir = "Saved Data/G_withDOI_v0"):
    # Load graph
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



def link_pred_jaccard(save_dir = "Saved Data/G_withDOI_v0/", APP_process=True):
    df = pd.read_csv("../Data (Reformatted)/df_all.csv")
    all_mat = set(df[(df["mat"]!="unknown")]["mat"].dropna())
    all_app = set(df[(df["app"]!="unknown")]["app"].dropna())
    all_prod = set(df[(df["prod"]!="unknown")]["prod"].dropna())


    data_dir = "../Inference_Results/pythia-2.8B-MC/checkpoint-7080/"
    df_mat_app = pd.read_csv(data_dir + "mat_app_count.csv", index_col=0)


    if APP_process:
        removed_APPs = ["pore forming agent", "reinforced fibre", "superplasticizer"]
        df_mat_app = df_mat_app[(~df_mat_app["Application"].isin(removed_APPs))]

        for MAT in list(df_mat_app["Material"].unique()):
            df_mat_app.loc[(df_mat_app["Material"] == MAT) & (df_mat_app["Application"] == "fine aggregate"), "count"] = \
            int(df_mat_app.loc[(df_mat_app["Material"] == MAT) & (df_mat_app["Application"] == "fine aggregate"), "count"]) +\
            int(df_mat_app.loc[(df_mat_app["Material"] == MAT) & (df_mat_app["Application"] == "aggregate"), "count"])
            
            df_mat_app.loc[(df_mat_app["Material"] == MAT) & (df_mat_app["Application"] == "coarse aggregate"), "count"] = \
            int(df_mat_app.loc[(df_mat_app["Material"] == MAT) & (df_mat_app["Application"] == "coarse aggregate"), "count"]) +\
            int(df_mat_app.loc[(df_mat_app["Material"] == MAT) & (df_mat_app["Application"] == "aggregate"), "count"])
        df_mat_app = df_mat_app[(df_mat_app["Application"] != "aggregate")]
    

    df_mat_app.to_csv(data_dir + "new_mat_app_count.csv", index=False)




    mat_app_count = df_mat_app.pivot(index="Application", columns="Material", values="count")
    mat_app_norm_forAPP = mat_app_count.apply(lambda x: x/sum(x), axis = 1)



    df_mat_jaccard = pd.read_csv(save_dir + "MAT_Jaccard.csv")
    jaccard = df_mat_jaccard.pivot(index="material1", columns="material2", values="jaccard")



    new_mat_app = np.matmul(mat_app_norm_forAPP.to_numpy(), jaccard.to_numpy())
    df_new_mat_app = mat_app_norm_forAPP.copy(deep=True)
    df_new_mat_app.iloc[:,:] = new_mat_app


    heatmap_diff = df_new_mat_app - mat_app_norm_forAPP



    heatmap_diff_at0 = heatmap_diff[mat_app_norm_forAPP == 0]
    sns.set(rc={'figure.figsize':(20, 3)})
    heatmap = sns.heatmap(heatmap_diff_at0, cmap="viridis", cbar_kws={'shrink': 1.0}, vmax=0.06, 
                xticklabels=True, yticklabels=True)
    fig = heatmap.get_figure()
    fig.savefig("../Figs/Graph/new_mat_app_link_pred_matSim.jpg", bbox_inches="tight")