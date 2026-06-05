import json
import pandas as pd
from clustering import run_kmeans
from feature_analysis import extract_radon_features, extract_ast_features, extract_all_features, extract_lexical_features
from GraphCodeBert import load_embeddings
from clustering import cluster_embeddings_per_puzzle, get_feature_columns, cluster_embeddings_hdbscan
from clusteringanalysis import within_puzzle_zscore, run_handcrafted_clustering_analysis, compare_clusterings
from persona_aggregation import aggregate_local_clusters_to_personas
import numpy as np
from efficiency_analysis import run_efficiency_analysis
import warnings

def main():
    warnings.simplefilter("ignore", category=FutureWarning) # to stop hundreds of warnings that otherwise print
    extract_all_features()
    with open('data/combined_features.json', 'r') as f:
        combined_data = json.load(f)
    combined_df = pd.DataFrame(combined_data)
    with open('data/astandradon_features.json', 'r') as f:
        astandradon_data = json.load(f)
    astandradon_df = pd.DataFrame(astandradon_data)
    combined_df = pd.DataFrame(combined_data)
    
    clustered_dfk2, consistency_dfk2 = run_handcrafted_clustering_analysis(
        df=combined_df,
        n_clusters=2,
        min_solutions=10,
        min_group_size=15
    )
    clustered_dfk4, consistency_dfk4 = run_handcrafted_clustering_analysis(
        df=combined_df,
        n_clusters=4,
        min_solutions=10,
        min_group_size=15
    )
    clustered_dfastradon1, consistency_dfastradon1 = run_handcrafted_clustering_analysis(
        df=astandradon_df,
        n_clusters=2,
        min_solutions=10,
        min_group_size=15
    )
    clustered_dfastradon2, consistency_dfastradon2 = run_handcrafted_clustering_analysis(
        df=astandradon_df,
        n_clusters=4,
        min_solutions=10,
        min_group_size=15
    )
    df_emb = load_embeddings('data/graphcodebert_embeddings.json')
    df_local = cluster_embeddings_per_puzzle(df_emb, min_puzzle_size=15, min_cluster_size=5, min_samples=3, n_reduce=10)

    with open('data/combined_features.json') as f:
        df_feat = pd.DataFrame(json.load(f))
        features = get_feature_columns(df_feat)
        df_norm, _ = within_puzzle_zscore(df_feat, features, min_group_size=15)

    profilesk2, persona_labelsk2 = aggregate_local_clusters_to_personas(
        df_local=df_local, df_norm=df_norm, features=features,
        k_range=range(2, 9), final_k=2)
    profilesk4, persona_labels = aggregate_local_clusters_to_personas(
        df_local=df_local, df_norm=df_norm, features=features,
        k_range=range(2, 9), final_k=4)
    
    matched = run_efficiency_analysis(
        clustered_dfk2,
        'data/final_data_v1.json',
    )
    matched = run_efficiency_analysis(
        clustered_dfk4,
        'data/final_data_v1.json',
    )
if __name__ == "__main__":
    main()