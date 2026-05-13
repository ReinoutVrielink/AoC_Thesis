import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap
import hdbscan
 
def get_feature_columns(df):
    # Extract all numeric feature columns, excluding metadata that is not relevant for clustering
    metadata_cols = {'sol_id', 'author', 'year', 'day', 'part', 'code', 'language', 'puzzle'}
    feature_cols = [col for col in df.columns if col not in metadata_cols and df[col].dtype in ['float64', 'int64']]
    return feature_cols
 
 
def run_kmeans(df, n_clusters=2, features=None):
    df = df.copy()
    if features is None:
        features = get_feature_columns(df)
    # Remove rows with missing values in selected features
    df = df.dropna(subset=features)
    # scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    # 5/5/2026: new - find optimal k using silhouette score
    # added code is from https://farshadabdulazeez.medium.com/understanding-silhouette-score-in-clustering-8aedc06ce9c4
    scores = []
    k_range = range(2, 10)

    for k in k_range:
        kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans_tmp.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)

    plt.figure()
    plt.plot(k_range, scores, marker='o')
    plt.title("Silhouette Score vs. Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.show()
    # running k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    # Silhouette score
    sil_score = silhouette_score(X_scaled, df['cluster'])
    print(f"Silhouette Score: {sil_score:.3f}")
    print(f"Features used: {len(features)}")
    print(f"Samples clustered: {len(df)}")
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    df_pca['cluster'] = df['cluster'].values
    plt.figure(figsize=(10, 7))
    # generated visualization with Gemini, to make visualization more clear and informative
    sns.scatterplot(
        x='PC1', y='PC2', hue='cluster',
        data=df_pca, palette='viridis',
        alpha=0.6, s=100
    )
    plt.title(f'K-means Clustering - {len(features)} features, Silhouette: {sil_score:.3f}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.tight_layout()
    plt.show()
    return df


def cluster_embeddings_hdbscan(df, min_cluster_size=10, min_samples=3, n_reduce=10, visualize=True):
    # This function clusters GraphCodeBERT embeddings with HDBSCAN and does UMAP dimensionality reduction
    # Two separate UMAP projections are used: one for clustering (n_reduce dims, tight packing)
    # and one for 2D visualization
    df = df.copy()
    X = np.array([emb for emb in df['embedding']])
    # normalizing the embeddings to focus on direction of data rather than magnitude
    X = normalize(X, norm='l2')

    # UMAP reducer for clustering: squash the many dimensions in the embeddings to 10 dimensions
    # min_dist=0.0 allows points to pack tightly so HDBSCAN can find dense regions
    reducer_cluster = umap.UMAP(
        n_components=n_reduce,
        n_neighbors=100,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
    )
    X_reduced = reducer_cluster.fit_transform(X)

    # Separate UMAP reducer for 2D visualization only.
    # Not used for clustering itself, since 2D loses too much info for HDBSCAN
    reducer_viz = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
    )
    X_umap = reducer_viz.fit_transform(X)
    print("Clustering with HDBSCAN.....")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    df['cluster'] = clusterer.fit_predict(X_reduced)
    # Cluster count excludes the outliers that HDBSCAN assigns
    n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'].values else 0)
    n_outliers = (df['cluster'] == -1).sum()
    print(f"\nNumber of clusters: {n_clusters}")
    print(f"Number of outliers: {n_outliers} / {len(df)}")
    print(f"\nCluster distribution:")
    print(df['cluster'].value_counts().sort_index())
    # Compute silhouette score only on non-outlier points, and only if there are
    # at least 2 clusters and 2 points (silhouette is undefined otherwise)
    mask = df['cluster'] != -1
    if mask.sum() > 1 and df.loc[mask, 'cluster'].nunique() > 1:
        sil_score = silhouette_score(X_reduced[mask], df.loc[mask, 'cluster'])
        print(f"Silhouette Score (excluding outliers): {sil_score:.3f}")
    # Plot the 2D UMAP projection coloured by cluster assignment.
    if visualize:
        # generated visualization with Gemini, to make visualization more clear and informative
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            x=X_umap[:, 0], y=X_umap[:, 1],
            hue=df['cluster'].astype(str),
            palette='viridis', alpha=0.6, s=100,
        )
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title('GraphCodeBERT Embeddings - HDBSCAN Clustering')
        plt.tight_layout()
        plt.show()
    return df
