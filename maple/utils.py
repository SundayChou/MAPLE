import os
import torch
import random
import subprocess
import numpy as np
import scanpy as sc
import anndata as ad


def fix_seed(seed=42):
    print(f'Global random seed set to: {seed}.')

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def scipy_to_torch_sparse(scipy_mat):
    crow_indices = torch.from_numpy(scipy_mat.indptr)
    col_indices = torch.from_numpy(scipy_mat.indices)
    values = torch.from_numpy(scipy_mat.data)
    size = scipy_mat.shape

    return torch.sparse_csr_tensor(crow_indices, col_indices, values, size)


def optimize_leiden(adata, n_clusters=10, used_obsm='emb', add_obs='leiden', res_min=0.01, res_max=2.0, max_tries=20, seed=42):
    best_labels = None
    tmp_adata = ad.AnnData(np.zeros((adata.shape[0], 1)))
    tmp_adata.obsm['X_feat'] = adata.obsm[used_obsm]
    sc.pp.neighbors(tmp_adata, use_rep='X_feat', n_neighbors=20, random_state=seed)

    for i in range(max_tries):
        res = (res_min + res_max) / 2
        sc.tl.leiden(tmp_adata, resolution=res, random_state=seed, directed=False, n_iterations=2, flavor='igraph')
        current_clusters = len(tmp_adata.obs['leiden'].cat.categories)

        print(f'Current resolution is {res:.4f}, found {current_clusters} clusters.')

        if current_clusters == n_clusters:
            best_labels = tmp_adata.obs['leiden'].values
            print(f'Success: Leiden found {n_clusters} clusters at resolution {res:.4f} (Attempt {i+1}).')
            break
        elif current_clusters < n_clusters:
            res_min = res
        else:
            res_max = res

    if best_labels is None:
        best_labels = tmp_adata.obs['leiden'].values
        print(f'Warning: Leiden failed to find exactly {n_clusters} clusters (found {current_clusters} clusters at resolution {res:.4f}).')

    adata.obs[add_obs] = np.array(best_labels).astype(int)
    adata.obs[add_obs] = adata.obs[add_obs].astype('category')


def idenfity_germinal_centers(adata, cluster_key, threshold=0.3):
    is_gc_bool = adata.obs['annotation'].astype(str) == 'Germinal Center'
    cluster_purity = is_gc_bool.groupby(adata.obs[cluster_key], observed=True).mean()
    gc_clusters = cluster_purity[cluster_purity > threshold].index

    adata.obs[cluster_key.replace('_leiden', '_is_GC')] = adata.obs[cluster_key].isin(gc_clusters)
    adata.obs[cluster_key.replace('_leiden', '_is_GC')] = adata.obs[cluster_key.replace('_leiden', '_is_GC')].astype('category')