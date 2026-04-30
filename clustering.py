import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
 
def run_kmeans(df, n_clusters=2, features=None):
    if features is None:
        features = [
            'source_code_lines', 'max_line_length', 'total_lines', 'unique_identifiers', 
            'volume', 'parenthesis_ratio', 'avg_string_literal_len', 'for_loop_count', 
            'avg_line_length', 'avg_identifier_length'
        ]
    df = df.copy().dropna(subset=features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    df_pca['cluster'] = df['cluster'].values
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=df_pca, palette='viridis', alpha=0.6)
    plt.title('Cluster Separation (PCA Reduced)')
    plt.show()
    sil_score = silhouette_score(X_scaled, df['cluster'])
    print(f"Silhouette Score: {sil_score:.3f}")
    
    return df