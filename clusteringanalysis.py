import json
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import Counter
from clustering import get_feature_columns
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, confusion_matrix

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
    df_norm, df_orig = within_puzzle_zscore(df, features, min_group_size=min_group_size)
    X = df_norm[features].values
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    print(f"\nK-means (k={n_clusters}): silhouette = {sil:.3f}")
    print(f"Cluster sizes: {sorted(Counter(labels).items())}")
    
    # attach labels to BOTH the original-scale and z-scored frames
    clustered_df = df_orig.copy()
    clustered_df['cluster'] = labels
    clustered_norm = df_norm.copy()
    clustered_norm['cluster'] = labels

    # PCA visualization
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels.astype(str), palette='viridis', alpha=0.5, s=30)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title(f'Global K-means clusters (k={n_clusters})')
    plt.tight_layout()
    plt.show()
    
    # cluster profiles on the within-puzzle z-scored scale (SMDs)
    pd.set_option('display.max_rows', None)
    profile_z = clustered_norm.groupby('cluster')[features].mean()
    print("\nCluster profiles (within-puzzle z-scored, standardised mean differences):")
    print(profile_z.T.round(2))
    
    consistency_df = author_consistency(clustered_df, min_solutions=min_solutions)
    
    return clustered_df, consistency_df

def compare_clusterings(kmeans_df, ward_df, label_col_kmeans='cluster', label_col_ward='global_cluster_id'):
    # This function takes the solutions that got both a K-means label and a Ward persona label, 
    # lines them up by sol_id, and measures how much the two clusterings agree using two standard scores 
    # (ARI and AMI, where 0 means random agreement and 1 means identical groupings)
    # It also prints a table showing how the K-means clusters map onto the Ward personas, so you can see where 
    # they agree rather than just by how much
    warnings.simplefilter("ignore", category=FutureWarning) # i dont want a sea of warnings before I run a function
    ward_df = ward_df[ward_df[label_col_ward] != -1]
    merged = kmeans_df[['sol_id', label_col_kmeans]].merge(
        ward_df[['sol_id', label_col_ward]], on='sol_id'
    )
    print(f"Solutions with both labels: {len(merged)}")
    a = merged[label_col_kmeans].values   # K-means labels
    b = merged[label_col_ward].values     # Ward persona labels
    print(f"ARI: {adjusted_rand_score(a, b):.3f}")
    print(f"AMI: {adjusted_mutual_info_score(a, b):.3f}")
    print("\nContingency table (rows = K-means, cols = Ward persona):")
    print(pd.crosstab(merged[label_col_kmeans], merged[label_col_ward]))
    return merged

def add_puzzle_column(df):
    # make one puzzle id out of year, day and part
    df = df.copy()
    df['puzzle'] = df['year'].astype(str) + '_' + df['day'].astype(str) + '_' + df['part'].astype(str)
    return df

def author_consistency_score(df, label_col, min_solutions):
    # different version of author_consistency for repeated calls in the shuffle loop
    # returns just (mean, n) instead of a DataFrame
    # used as a computational function that doesnt print anything
    scores = []
    for author, group in df.groupby('author'):
        if len(group) < min_solutions:
            continue
        most_common_label, count = Counter(group[label_col]).most_common(1)[0]
        scores.append(count / len(group))
    return np.mean(scores), len(scores)

def consistency_with_baseline(df, label_col, min_solutions=10, n_shuffle=100, seed=42):
    # real score, plus a shuffled baseline: we randomly swap the labels
    # around inside each puzzle and see what consistency we get by chance
    rng = np.random.default_rng(seed)
    df = add_puzzle_column(df)
    df = df[df['author'] != '[deleted]'].copy()
    real_score, n_authors = author_consistency_score(df, label_col, min_solutions)
    baselines = []
    for _ in range(n_shuffle):
        shuffled = df[label_col].copy()
        for _, idx in df.groupby('puzzle').groups.items():
            idx = list(idx)
            shuffled.loc[idx] = rng.permutation(df.loc[idx, label_col].values)
        df['shuffled'] = shuffled
        baseline_score, _ = author_consistency_score(df, 'shuffled', min_solutions)
        baselines.append(baseline_score)
    return real_score, n_authors, np.mean(baselines)

def puzzle_dominance(df, label_col):
    # for every puzzle, what share of its solutions are in the single
    # most common cluster/persona
    df = add_puzzle_column(df)
    doms = []
    for _, group in df.groupby('puzzle'):
        counts = group[label_col].value_counts()
        doms.append(counts.iloc[0] / len(group))
    doms = np.array(doms)
    return {
        'n_puzzles': len(doms),
        'mean': doms.mean(),
        'median': np.median(doms),
        'max': doms.max(),
        'n_above_080': int((doms > 0.80).sum()),
    }