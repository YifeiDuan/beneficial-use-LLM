import networkx as nx
import math
import numpy as np
import pandas as pd
import os


if __name__ == "__main__":

    # Read in tabular data of all text mining entries
    df = pd.read_csv("../Data (Reformatted)/df_all.csv")


    all_mat = set(df[(df["mat"]!="unknown")]["mat"].dropna())
    all_app = set(df[(df["app"]!="unknown")]["app"].dropna())
    all_prod = set(df[(df["prod"]!="unknown")]["prod"].dropna())


    # Add nodes
    G = nx.Graph()

    for mat in all_mat:
        G.add_node(mat, node_type="material")
        
    for app in all_app:
        G.add_node(app, node_type="application")
        
    for prod in all_prod:
        if prod in nx.nodes(G):
            prod = prod + " product"
        G.add_node(prod, node_type="product")
    

    # Add edges
    for idx in range(len(df)):
        DOI = df["DOI"][idx]
        mat = df["mat"][idx]
        app = df["app"][idx]
        prod = df["prod"][idx]
        
        if not G.has_node(DOI):
            G.add_node(DOI, node_type="DOI")
            
        if G.has_node(mat):
            G.add_edge(DOI, mat, weight=1, edge_type="DOI_mat")
            
        if G.has_node(prod):
            G.add_edge(DOI, prod, weight=1, edge_type="DOI_prod")
        
        if G.has_node(mat) and G.has_node(app):
            if not G.has_edge(mat, app):
                G.add_edge(mat, app, weight=1, edge_type="mat_app")
            else:
                G[mat][app]["weight"] += 1
    

    # Save graph
    save_dir = "Saved Data/G_withDOI_v0/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    nx.write_graphml_lxml(G, save_dir + "Graph.graphml")