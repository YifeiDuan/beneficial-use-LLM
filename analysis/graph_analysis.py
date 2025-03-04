import numpy as np
import pandas as pd
import math
import os
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import random


def weighted_jaccard_similarity_by_type(G, node1, node2, target_types):
    """
    target_types is a sublist of ["DOI", "material", "application", "product"]
    """
    # Get neighbors of both nodes
    neighbors1 = set(G.neighbors(node1))
    neighbors2 = set(G.neighbors(node2))

    # Filter neighbors by type
    neighbors1_of_type = {n for n in neighbors1 if G.nodes[n].get('node_type') in target_types}
    neighbors2_of_type = {n for n in neighbors2 if G.nodes[n].get('node_type') in target_types}
    
    # Calculate total neighbor edge weights (for further normalization)
    total_weights_1 = sum(G[node1][n].get('weight', 1.0) for n in neighbors1_of_type)
    total_weights_2 = sum(G[node2][n].get('weight', 1.0) for n in neighbors2_of_type)

    # Find the intersection and union of neighbors of the target type
    common_neighbors = neighbors1_of_type & neighbors2_of_type
    all_neighbors = neighbors1_of_type | neighbors2_of_type

    # If no common neighbors of the given type, similarity is 0
    if not common_neighbors:
        return 0

    # Initialize sums for minimum and maximum weights
    min_weight_sum = 0
    max_weight_sum = 0
    
    # Loop through common neighbors of the specified type
    for neighbor in common_neighbors:
        w1 = G[node1][neighbor].get('weight', 1.0) / total_weights_1  # Normalized weight from node1 to neighbor
        w2 = G[node2][neighbor].get('weight', 1.0) / total_weights_2  # Normalized weight from node2 to neighbor
        min_weight_sum += min(w1, w2)
    
    # Loop through all neighbors of the specified type for maximum weight calculation
    for neighbor in all_neighbors:
        w1 = G[node1].get(neighbor, {}).get('weight', 0) / total_weights_1  # Default to 0 if no edge
        w2 = G[node2].get(neighbor, {}).get('weight', 0) / total_weights_2
        max_weight_sum += max(w1, w2)

    # Compute Weighted Jaccard Similarity
    weighted_jaccard = min_weight_sum / max_weight_sum if max_weight_sum != 0 else 0
    
    return weighted_jaccard




def perturb_edge_weight(G, u, v):
    current_weight = G[u][v]['weight'] if G.has_edge(u, v) else 0
    if current_weight == 0:
        G.add_edge(u, v, weight=current_weight + 1)
    else:
        G.add_edge(u, v, weight=current_weight + random.choice([-1,1]))
    return G



def perturb_graph_monte_carlo(G, p_threshold=0.8):
    G_sampled = G.copy()

    mat_nodes = [node for node, data in G.nodes(data=True) if data.get("node_type") == "material"]
    app_nodes = [node for node, data in G.nodes(data=True) if data.get("node_type") == "application"]

    for mat in mat_nodes:
        for app in app_nodes:
            random_p = random.uniform(0,1)
            if random_p < p_threshold:
                continue
            else:   # IF [0,1)-uniformly-sampled random_p >= p_threshold
                G_sampled = perturb_edge_weight(G_sampled, mat, app)  # THEN perturb the edge weight by 1 or -1
    
    return G_sampled
        



def calculate_weighted_jaccard_perturb(save_dir = "Saved Data/G_withDOI_v0/", target_types=["application"], num_samples=100, p_threshold=0.8):
    # Read data
    G = nx.read_graphml(save_dir + "Graph.graphml")

    df = pd.read_csv("../Data (Reformatted)/df_all.csv")

    all_mat = set(df[(df["mat"]!="unknown")]["mat"].dropna())
    all_app = set(df[(df["app"]!="unknown")]["app"].dropna())
    all_prod = set(df[(df["prod"]!="unknown")]["prod"].dropna())



    mat_nodes = [node for node, data in G.nodes(data=True) if data.get("node_type") == "material"]
    mat_pairs = []
    for mat1 in mat_nodes:
        for mat2 in mat_nodes:
            mat_pairs.append((mat1, mat2))

    
    for i in range(num_samples):
        # Perturb the graph
        G_sampled = perturb_graph_monte_carlo(G, p_threshold=p_threshold)
                    
        # Calculate weighted jaccard
        mat_jaccard_records = []

        target_types = ["application"]
        for (mat1, mat2) in mat_pairs:
            record = {
                "material1": mat1,
                "material2": mat2,
                "jaccard":   weighted_jaccard_similarity_by_type(G_sampled, mat1, mat2, target_types)
            }
            mat_jaccard_records.append(record)

        df_mat_jaccard = pd.DataFrame.from_records(mat_jaccard_records)

        df_mat_jaccard.to_csv(save_dir + f"MAT_Jaccard_weighted_perturbed_{i}.csv", index=False)



        # Plot
        data = df_mat_jaccard.pivot(index="material1", columns="material2", values="jaccard")
        ##### Set mask
        lower_triangle_indices = np.tril_indices(data.shape[0])
        mask = np.ones(data.shape, dtype=bool)
        mask[lower_triangle_indices] = False

        plt.figure()
        sns.set(rc={'figure.figsize':(15, 12)})
        heatmap = sns.heatmap(data, vmin=0, vmax=1, 
                            cmap="hot",
                            cbar_kws={'shrink': 0.3,
                                        'label': "Jaccard Coefficient (Node Similarity)"}, 
                            mask=mask,
                            xticklabels=True, yticklabels=True)

        fig = heatmap.get_figure()

        fig_dir = "../Figs/Graph & Link Pred/perturbed/"
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        fig.savefig(fig_dir + f"mat_jaccard_weighted_heatmap_lower_triangle_perturbed_{i}.jpg", bbox_inches="tight")



def link_pred_jaccard_uq(save_dir = "Saved Data/G_withDOI_v0/", num_samples=100):

    data_dir = "../Inference_Results_Analysis_df/pythia-2.8B-MC/checkpoint-7080/"
    df_mat_app = pd.read_csv(data_dir + "new_mat_app_count.csv", index_col=0)


    mat_app_count = df_mat_app.pivot(index="Application", columns="Material", values="count")
    mat_app_norm_forAPP = mat_app_count.apply(lambda x: x/sum(x), axis = 1)

    link_pred_perturbed_list = []

    for i in range(num_samples):
        df_mat_jaccard = pd.read_csv(save_dir + f"MAT_Jaccard_weighted_perturbed_{i}.csv")
        jaccard = df_mat_jaccard.pivot(index="material1", columns="material2", values="jaccard")



        new_mat_app = np.matmul(mat_app_norm_forAPP.to_numpy(), jaccard.to_numpy())
        df_new_mat_app = mat_app_norm_forAPP.copy(deep=True)
        df_new_mat_app.iloc[:,:] = new_mat_app


        heatmap_diff = df_new_mat_app - mat_app_norm_forAPP


        heatmap_diff_at0 = heatmap_diff[mat_app_norm_forAPP == 0]
        heatmap_diff_at0.to_csv(data_dir + f"link_pred_jaccard_weighted_perturbed_{i}.csv")

        link_pred_perturbed_list.append(heatmap_diff_at0)
    

    # Compute cell-wise mean and std for the link predictions
    link_pred_mean = sum(link_pred_perturbed_list) / len(link_pred_perturbed_list)
    link_pred_std = link_pred_mean.copy(deep=True)
    link_pred_std.iloc[:,:] = np.std(np.array([df.values for df in link_pred_perturbed_list]), axis=0, ddof=0)

    # Plot mean
    plt.figure()
    sns.set(rc={'figure.figsize':(20, 3)})
    heatmap = sns.heatmap(link_pred_mean, cmap="hot", cbar_kws={'shrink': 1.0,
                                                                'label': "Predicted Link Score"}, 
                        vmax=1.0, 
                        xticklabels=True, yticklabels=True)

    heatmap.collections[0].colorbar.set_ticks([0.2, 0.4, 0.6, 0.8, 1.0])
    heatmap.collections[0].colorbar.set_ticklabels(["0.200", "0.400", "0.600", 
                                                    "0.800", "1.000"])

    fig = heatmap.get_figure()
    fig.savefig("../Figs/Graph & Link Pred/new_mat_app_link_pred_matSim_perturbed_mean.jpg", bbox_inches="tight")


    # Plot std
    plt.figure()
    sns.set(rc={'figure.figsize':(20, 3)})
    heatmap = sns.heatmap(link_pred_std, cmap="hot", cbar_kws={'shrink': 1.0,
                                                                'label': "Predicted Link Score Uncertainty"}, 
                        vmax=0.12, 
                        xticklabels=True, yticklabels=True)

    heatmap.collections[0].colorbar.set_ticks([0.02, 0.04, 0.06, 0.08, 0.10])
    heatmap.collections[0].colorbar.set_ticklabels(["0.020", "0.040", "0.060", "0.080", "0.100"])

    fig = heatmap.get_figure()
    fig.savefig("../Figs/Graph & Link Pred/new_mat_app_link_pred_matSim_perturbed_std.jpg", bbox_inches="tight")



def link_pred_subplot(data_dir = "../Inference_Results_Analysis_df/pythia-2.8B-MC/checkpoint-7080/"):
    ### Read data
    link_pred_mean = pd.read_csv(data_dir + "link_pred_jaccard_weighted_mean.csv",  index_col=0)
    link_pred_std  = pd.read_csv(data_dir + "link_pred_jaccard_weighted_std.csv",  index_col=0)


    ### Select APPs and MATs
    link_pred_mean = link_pred_mean.drop(["additive"])
    link_pred_std  = link_pred_std.drop(["additive"])

    row = link_pred_mean.loc["lime-pozzolan cement"]

    MATs_g1 = list(row[row>0.55].keys())
    unwanted = ["lime", "fly ash", "cellulose", "silica", "natural pozzolan"]
    for MAT in unwanted:
        if MAT in MATs_g1:
            MATs_g1.remove(MAT)
    MATs_g2 = ["MSWI fly ash", "coal fly ash", "blast furnace slag", "silica fume", "nano-silica", "biomass fly ash", 
            "copper tailing", "coal bottom ash", "waste glass", "basalt fiber", "oil palm shell"]

    MATs = list(set(MATs_g1 + MATs_g2))
    MATs.sort()

    link_pred_mean = link_pred_mean[MATs]
    link_pred_std  = link_pred_std[MATs]
    

    ### Sort columns
    sorted_columns = link_pred_mean.loc['lime-pozzolan cement'].sort_values(ascending=False).index

    link_pred_mean = link_pred_mean[sorted_columns]
    link_pred_std  = link_pred_std[sorted_columns]


    ### Plot
    ##### Mean #####
    plt.figure()
    sns.set(rc={'figure.figsize':(9, 3)})
    heatmap = sns.heatmap(link_pred_mean, cmap="hot", cbar_kws={'shrink': 1.0,
                                                                    'label': "Predicted Link Score",
                                                                    #'location': "top"
                                                                    }, 
                        vmax=1, 
                        xticklabels=True, yticklabels=True)

    heatmap.get_yticklabels()[-3].set_fontweight("bold")

    heatmap.collections[0].colorbar.set_ticks([0.2, 0.4, 0.6, 0.8, 1.0])
    heatmap.collections[0].colorbar.set_ticklabels(["0.200", "0.400", "0.600", 
                                                    "0.800", "1.000"])

    fig = heatmap.get_figure()
    fig.savefig("../Figs/Graph & Link Pred/new_mat_app_link_pred_weighted_jaccard_sub_perturbed_mean.jpg", bbox_inches="tight", dpi=300)

    ##### std #####
    plt.figure()
    sns.set(rc={'figure.figsize':(9, 3)})
    heatmap = sns.heatmap(link_pred_std, cmap="hot", cbar_kws={'shrink': 1.0,
                                                            'label': "Predicted Link Score Uncertainty"}, 
                      vmax=0.12, 
                      xticklabels=True, yticklabels=True)

    heatmap.get_yticklabels()[-3].set_fontweight("bold")

    heatmap.collections[0].colorbar.set_ticks([0.02, 0.04, 0.06, 0.08, 0.10])
    heatmap.collections[0].colorbar.set_ticklabels(["0.020", "0.040", "0.060", "0.080", "0.100"])

    fig = heatmap.get_figure()
    fig.savefig("../Figs/Graph & Link Pred/new_mat_app_link_pred_weighted_jaccard_sub_perturbed_mean.jpg", bbox_inches="tight", dpi=300)








def calculate_weighted_jaccard(save_dir = "Saved Data/G_withDOI_v0/", target_types=["application"]):
    # Read data
    save_dir = "Saved Data/G_withDOI_v0/"
    G = nx.read_graphml(save_dir + "Graph.graphml")

    df = pd.read_csv("../Data (Reformatted)/df_all.csv")

    all_mat = set(df[(df["mat"]!="unknown")]["mat"].dropna())
    all_app = set(df[(df["app"]!="unknown")]["app"].dropna())
    all_prod = set(df[(df["prod"]!="unknown")]["prod"].dropna())



    mat_nodes = [node for node, data in G.nodes(data=True) if data.get("node_type") == "material"]
    mat_pairs = []
    for mat1 in mat_nodes:
        for mat2 in mat_nodes:
            mat_pairs.append((mat1, mat2))

    
    # Calculate weighted jaccard
    mat_jaccard_records = []

    target_types = ["application"]
    for (mat1, mat2) in mat_pairs:
        record = {
            "material1": mat1,
            "material2": mat2,
            "jaccard":   weighted_jaccard_similarity_by_type(G, mat1, mat2, target_types)
        }
        mat_jaccard_records.append(record)

    df_mat_jaccard = pd.DataFrame.from_records(mat_jaccard_records)

    df_mat_jaccard.to_csv(save_dir + "MAT_Jaccard_weighted.csv", index=False)



    # Plot
    data = df_mat_jaccard.pivot(index="material1", columns="material2", values="jaccard")
    ##### Set mask
    lower_triangle_indices = np.tril_indices(data.shape[0])
    mask = np.ones(data.shape, dtype=bool)
    mask[lower_triangle_indices] = False

    sns.set(rc={'figure.figsize':(15, 12)})
    heatmap = sns.heatmap(data, vmin=0, vmax=1, 
                        cmap="hot",
                        cbar_kws={'shrink': 0.3,
                                    'label': "Jaccard Coefficient (Node Similarity)"}, 
                        mask=mask,
                        xticklabels=True, yticklabels=True)

    fig = heatmap.get_figure()

    fig_dir = "../Figs/Graph & Link Pred/"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    fig.savefig(fig_dir + "mat_jaccard_weighted_heatmap_lower_triangle.jpg", bbox_inches="tight")



def link_pred_jaccard(save_dir = "Saved Data/G_withDOI_v0/", APP_process=True):
    df = pd.read_csv("../Data (Reformatted)/df_all.csv")
    all_mat = set(df[(df["mat"]!="unknown")]["mat"].dropna())
    all_app = set(df[(df["app"]!="unknown")]["app"].dropna())
    all_prod = set(df[(df["prod"]!="unknown")]["prod"].dropna())


    data_dir = "../Inference_Results_Analysis_df/pythia-2.8B-MC/checkpoint-7080/"
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
