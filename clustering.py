import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
 
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
    # Visualization
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=df_pca, palette='viridis', alpha=0.6, s=100)
    plt.title(f'K-means Clustering - {len(features)} features, Silhouette: {sil_score:.3f}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.tight_layout()
    plt.show()
    
    return df