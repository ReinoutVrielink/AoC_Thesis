import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, cophenet
from scipy.spatial.distance import pdist

# clustering the graphcodebert embeddings per puzzle gives us many small local clusters
# in this code we describe each of these local clusters by the mean of its data points within-puzzle z-scored features
# we cluster those profiles with Ward hierarchical clustering to find stylistic groups across puzzles
# most of code was realized by following this tutorial: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

def aggregate_local_clusters_to_personas(df_local, df_norm, features, k_range=range(2, 9), final_k=4):
    # drop outliers, attach z-scored features to local cluster labels
    df_local = df_local[df_local['global_cluster_id'] != -1]
    merged = df_local[['sol_id', 'global_cluster_id']].merge(
        df_norm[['sol_id'] + features], on='sol_id'
    )
    # one mean-profile per local cluster
    profiles = merged.groupby('global_cluster_id')[features].mean()
    sizes = merged.groupby('global_cluster_id').size()
    print(f"Built profiles for {len(profiles)} local clusters "
          f"covering {sizes.sum()} solutions")
    # K-means sweep for silhouette comparison only (keeps the sweep we had before
    # so we can compare both methods)
    # commented it out after I got the results
    #print("\nK-means silhouette scores + balance on cluster profiles:")
    #print("k, silhouette, share of biggest cluster")
    #for k in k_range:
    #    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(profiles)
    #    sil = silhouette_score(profiles, labels)
    #    sols_per_cluster = pd.Series(sizes.values).groupby(labels).sum()
    #    max_share = sols_per_cluster.max() / sols_per_cluster.sum()
    #    print(k, round(sil, 3), round(max_share, 3))

    # build the linkage matrix with Ward's method
    # taken from https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    Z = linkage(profiles.values, method='ward')

    # cophenetic correlation: how well the dendrogram preserves the original
    # pairwise distances between profiles. Closer to 1 = more faithful.
    # source is again https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    coph_corr, _ = cophenet(Z, pdist(profiles.values))
    print(f"\nCophenetic correlation: {coph_corr:.3f}")

    print("\nWard silhouette scores + balance on cluster profiles:")
    print("k, silhouette, share of biggest cluster")
    for k in k_range:
        ward_labels = fcluster(Z, t=k, criterion='maxclust')
        sil = silhouette_score(profiles, ward_labels)
        sols_per_cluster = pd.Series(sizes.values).groupby(ward_labels).sum()
        max_share = sols_per_cluster.max() / sols_per_cluster.sum()
        print(k, round(sil, 3), round(max_share, 3))

    # show the last few merge distances to spot any big "jump" that suggests a
    # natural cut-off
    # source: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    print(f"\nLast 6 merge distances:")
    print(Z[-6:, 2].round(2))

    # reducing the dendogram
    # source for this figure https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    plt.figure(figsize=(12, 5))
    dendrogram(
        Z,
        truncate_mode='lastp',
        p=12,
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
    )
    plt.title('Hierarchical clustering of local-cluster profiles (Ward)')
    plt.xlabel('local cluster or (cluster size)')
    plt.ylabel('distance')
    plt.tight_layout()
    plt.show()

    # final persona assignment by cutting the tree at final_k clusters
    # code also taken from https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    persona_labels = fcluster(Z, t=final_k, criterion='maxclust')
    # describing each persona by its features that deviate the most
    profiles = profiles.copy()
    profiles['persona'] = persona_labels
    print(f"\nFinal Ward partition at k={final_k}:")
    for persona, group in profiles.groupby('persona'):
        n_sols = sizes[group.index].sum()
        means = group[features].mean()
        top = means.reindex(means.abs().sort_values(ascending=False).index).head(8)
        print(f"\nPersona {persona}  (n_local_clusters={len(group)}, n_solutions={n_sols})")
        print(top.round(2).to_string())

    return profiles, persona_labels