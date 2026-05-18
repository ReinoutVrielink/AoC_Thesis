import json
import pandas as pd
from clustering import run_kmeans
from feature_analysis import extract_radon_features, extract_ast_features, extract_all_features, extract_lexical_features
from GraphCodeBert import load_embeddings
from clustering import cluster_embeddings_per_puzzle, get_feature_columns
from clusteringanalysis import within_puzzle_zscore, run_handcrafted_clustering_analysis
from persona_aggregation import aggregate_local_clusters_to_personas
import numpy as np

def main():
    extract_all_features()
    with open('data/combined_features.json', 'r') as f:
        combined_data = json.load(f)
    combined_df = pd.DataFrame(combined_data)
    print("\nStep 2: Running clustering analysis pipeline...")
    clustered_df, consistency_df = run_handcrafted_clustering_analysis(
        df=combined_df,
        n_clusters=2,
        min_solutions=10,
        min_group_size=10
    )
    df_emb = load_embeddings('data/graphcodebert_embeddings.json')
    df_emb['embedding'] = df_emb['embedding'].apply(lambda x: np.array(x))
    df_local = cluster_embeddings_per_puzzle(
    df_emb, min_puzzle_size=15, min_cluster_size=5, min_samples=3,)
    with open('data/combined_features.json') as f:
        df_feat = pd.DataFrame(json.load(f))
        features = get_feature_columns(df_feat)
        df_norm, _ = within_puzzle_zscore(df_feat, features)
    profiles, persona_labels = aggregate_local_clusters_to_personas(
        df_local=df_local,
        df_norm=df_norm,
        features=features,
        k_range=range(2, 9),
        final_k=4,
        )
    profiles, persona_labels = aggregate_local_clusters_to_personas(
        df_local=df_local,
        df_norm=df_norm,
        features=features,
        k_range=range(2, 9),
        final_k=2,
        )
if __name__ == "__main__":
    main()