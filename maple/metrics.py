import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors


def calculate_icc(adata, feat_key, cluster_key):
    X = np.asarray(adata.obsm[feat_key])
    labels = np.asarray(adata.obs[cluster_key])

    unique_classes, group_counts = np.unique(labels, return_counts=True)
    cluster_counts = pd.Series(group_counts, index=unique_classes)
    cluster_ratio = cluster_counts / cluster_counts.sum()
    dummies = pd.get_dummies(labels)

    global_mean = np.mean(X, axis=0)
    global_var = np.var(X, axis=0)
    SST = np.sum((X - global_mean) ** 2, axis=0)

    cicc_dict = {}
    group_means = []

    for c in unique_classes:
        X_c = X[labels == c]
        group_means.append(np.mean(X_c, axis=0))
        cluster_var = np.var(X_c, axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            cicc_per_feature = np.nan_to_num(1.0 - (cluster_var / global_var), nan=0.0)
            cicc_per_feature = np.clip(cicc_per_feature, 0.0, 1.0)
        cicc_dict[c] = np.mean(cicc_per_feature)

    group_means = np.array(group_means)
    SSB = np.sum(group_counts[:, None] * (group_means - global_mean) ** 2, axis=0)

    with np.errstate(invalid='ignore', divide='ignore'):
        ficc_per_feature = np.nan_to_num(SSB / SST, nan=0.0)
    ficc = np.mean(ficc_per_feature)

    cicc_df = pd.DataFrame({
        "CICC": pd.Series(cicc_dict),
        "Ratio": cluster_ratio
    }, index=dummies.columns).rename_axis('Cluster').sort_values(by="CICC", ascending=False)
    gicc = (cicc_df["CICC"] * cicc_df["Ratio"]).sum()

    return ficc, cicc_df, gicc


def calculate_js(adata, feat_key, emb_key, k=30):
    original_X = adata.obsm[feat_key]
    latent_X = adata.obsm[emb_key]

    idx_orig = NearestNeighbors(n_neighbors=k+1).fit(original_X).kneighbors(original_X, return_distance=False)[:, 1:]
    idx_latent = NearestNeighbors(n_neighbors=k+1).fit(latent_X).kneighbors(latent_X, return_distance=False)[:, 1:]

    jaccard_scores = []
    for orig, latent in zip(idx_orig, idx_latent):
        intersection = len(set(orig) & set(latent))
        union = 2 * k - intersection
        score = intersection / union if union > 0 else 0
        jaccard_scores.append(score)

    target_obs_key = f'{emb_key}_js_k={k}'
    adata.obs[target_obs_key] = jaccard_scores
    global_js = np.mean(jaccard_scores)

    return global_js