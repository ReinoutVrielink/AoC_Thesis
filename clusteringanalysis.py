import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import Counter
from clustering import get_feature_columns


def within_puzzle_zscore(df, features, min_group_size=10):
    # Z-score each feature within its (year, day, part) group
    df = df.dropna(subset=features).copy()
    group_sizes = df.groupby(['year', 'day', 'part']).size()
    df = df[df.set_index(['year', 'day', 'part']).index.map(group_sizes) >= min_group_size].copy()
    print(f"Kept {len(df)} solutions in groups >= {min_group_size}")
    global_std = df[features].std()
    normalized = df.copy()
    for _, group in df.groupby(['year', 'day', 'part']):
        means = group[features].mean()
        stds = group[features].std().replace(0, np.nan).fillna(global_std)
        normalized.loc[group.index, features] = (group[features] - means) / stds
    normalized[features] = normalized[features].fillna(0)
    return normalized, df


def author_consistency(df_with_labels, min_solutions=10):
    # in this thesis we are making coding personas
    # if these coding personas are meaningful, we would expect authors to fall 
    # into the same cluster across their different solutions somewhat consistently
    df = df_with_labels[df_with_labels['author'] != '[deleted]']
    records = []
    for author, group in df.groupby('author'):
        if len(group) < min_solutions:
            continue
        modal, modal_count = Counter(group['cluster']).most_common(1)[0]
        records.append({
            'author': author,
            'n_solutions': len(group),
            'modal_cluster': modal,
            'consistency': modal_count / len(group)
        })
    consistency_df = pd.DataFrame(records)
    print(f"\nAuthors with a minimum of {min_solutions} solutions: {len(consistency_df)}")
    print(f"Mean consistency in getting assigned to same cluster : {consistency_df['consistency'].mean():.3f}")
    return consistency_df


def run_handcrafted_clustering_analysis(df, n_clusters=2, min_solutions=10, min_group_size=10):
    features = get_feature_columns(df)
    print(f"Loaded {len(df)} solutions, {len(features)} features\n")
    
    df_norm, df_orig = within_puzzle_zscore(df, features, min_group_size=min_group_size)
    X = df_norm[features].values
    
    # i dont use kmeans function from clustering.py here, because I want to do some additional analysis and visualization that is not in that function
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    print(f"\nK-means (k={n_clusters}): silhouette = {sil:.3f}")
    print(f"Cluster sizes: {sorted(Counter(labels).items())}")
    
    clustered_df = df_orig.copy()
    clustered_df['cluster'] = labels

    # PCA visualization, made by Claude
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels.astype(str),
                    palette='viridis', alpha=0.5, s=30)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title(f'Global K-means clusters (k={n_clusters})')
    plt.tight_layout()
    plt.show()
    
    # cluster profiles on original scale
    pd.set_option('display.max_rows', None)
    profile = clustered_df.groupby('cluster')[features].mean()
    print("\nCluster profiles (original feature values):")
    print(profile.T.round(2))
    
    consistency_df = author_consistency(clustered_df, min_solutions=min_solutions)
    
    return clustered_df, consistency_df


if __name__ == "__main__":
    # 2 was taken from earlier silhouette analysis
    N_CLUSTERS = 2  
    with open('data/combined_features.json', 'r') as f:
        df = pd.DataFrame(json.load(f))
    run_handcrafted_clustering_analysis(df, n_clusters=N_CLUSTERS)