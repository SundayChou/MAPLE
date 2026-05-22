import os
import re
import time
import scipy
import mygene
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from pyfaidx import Fasta
from pyjaspar import jaspardb
from goatools.obo_parser import GODag
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics.pairwise import cosine_similarity


def compute_pca(X, n_comps=20, random_state=42):
    pca = PCA(n_components=n_comps, svd_solver='randomized', random_state=random_state)

    if isinstance(X, sp.csr_matrix) or isinstance(X, sp.csc_matrix):
        X_pca = pca.fit_transform(X.toarray())
    else:
        X_pca = pca.fit_transform(X)

    return X_pca


def compute_lsi(adata, n_components=20):
    X = adata.X
    idf = np.asarray(X.shape[0] / X.sum(axis=0))
    X_idf = X.multiply(idf) if scipy.sparse.issparse(X) else X * idf
    X_norm = normalize(X_idf, norm="l1", copy=False)

    if sp.issparse(X_norm):
        X_norm.data = np.log1p(X_norm.data * 1e4)
    else:
        X_norm = np.log1p(X_norm * 1e4)

    X_lsi = randomized_svd(X_norm, n_components + 1)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_feat"] = X_lsi[:, 1:]


def clr_normalize_each_cell(adata):
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))

        return np.log1p(x / exp)

    adata.X = np.apply_along_axis(seurat_clr, 1, (adata.X.toarray() \
        if sp.issparse(adata.X) else np.array(adata.X)))


def extract_hvps_by_coarse_clustering(adata, n_top_peaks=200, res=1.0, seed=42):
    sc.pp.neighbors(adata, use_rep='X_feat', key_added='coarse_knn', n_neighbors=20, random_state=seed)
    sc.tl.leiden(adata, resolution=res, neighbors_key='coarse_knn', key_added='coarse_leiden', 
                 random_state=seed, directed=False, n_iterations=2, flavor='igraph')
    sc.tl.rank_genes_groups(adata, groupby='coarse_leiden', method='t-test', 
                            use_raw=False, key_added='coarse_rank_peaks')

    rank_names = adata.uns['coarse_rank_peaks']['names']
    hvp_set = set(np.concatenate([rank_names[c][:n_top_peaks] for c in rank_names.dtype.names]))
    adata.var['highly_variable'] = adata.var_names.isin(hvp_set)


def get_adapt_sigma(dist):
    nonzero = dist[dist > 0]
    adapt_sigma = max(np.median(nonzero) if len(nonzero) > 0 else 1.0, 1e-8)
    return adapt_sigma


def normalize_adj(adj, add_eye=True):
    adj = sp.csr_matrix(adj)
    if add_eye:
        adj = adj + sp.eye(adj.shape[0])

    d_inv_sqrt = np.power(np.array(adj.sum(1)).flatten(), -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    normalized_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocsr()

    return normalized_adj


def build_spot_graphs(adata, n_spat_neighbors=6, n_feat_neighbors=20):
    N = adata.shape[0]

    nbrs_spat = NearestNeighbors(n_neighbors=n_spat_neighbors + 1, metric='cityblock', n_jobs=-1).fit(adata.obsm['spatial'])
    dist_spat, idx_spat = nbrs_spat.kneighbors(adata.obsm['spatial'])
    current_spat_sigma = get_adapt_sigma(dist_spat)
    w_spat = np.exp(-(dist_spat ** 2) / (2 * (current_spat_sigma ** 2)))

    nbrs_feat = NearestNeighbors(n_neighbors=n_feat_neighbors, metric='correlation', n_jobs=-1).fit(adata.obsm['X_feat'])
    dist_feat, idx_feat = nbrs_feat.kneighbors(adata.obsm['X_feat'])
    current_feat_sigma = get_adapt_sigma(dist_feat)
    w_feat = np.exp(-(dist_feat ** 2) / (2 * (current_feat_sigma ** 2)))

    spat_adj_raw = sp.csr_matrix((w_spat.flatten(), (np.repeat(np.arange(N), n_spat_neighbors + 1), idx_spat.flatten())), shape=(N, N))
    feat_adj_raw = sp.csr_matrix((w_feat.flatten(), (np.repeat(np.arange(N), n_feat_neighbors), idx_feat.flatten())), shape=(N, N))
    joint_adj_raw = spat_adj_raw.maximum(feat_adj_raw.multiply(feat_adj_raw.T))

    spat_adj_norm = normalize_adj(spat_adj_raw, False).astype(np.float32)
    feat_adj_norm = normalize_adj(feat_adj_raw, False).astype(np.float32)
    joint_adj_norm = normalize_adj(joint_adj_raw, False).astype(np.float32)

    print('The spatial, feature, and joint graphs have been successfully generated.')

    return spat_adj_norm, feat_adj_norm, joint_adj_norm


def build_go_graph(adata, n_neighbors=5, dataset_type='real',
                   obo_path='../data/feature_prior/go-basic.obo',
                   graph_path='../data/feature_prior/go_graphs/', save_graph=True):
    if dataset_type == 'sim':
        N = adata.shape[1]
        rows, cols, data = [], [], []

        for i in range(N):
            choices = np.concatenate([np.arange(i), np.arange(i + 1, N)])
            selected_cols = np.random.choice(choices, n_neighbors, replace=False)

            rows.extend([i] * n_neighbors)
            cols.extend(selected_cols)
            data.extend([1e-8] * n_neighbors)

        rows.extend(range(N))
        cols.extend(range(N))
        data.extend([1.0] * N)

        go_adj_raw = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
        go_adj_norm = normalize_adj(go_adj_raw, False).astype(np.float32)

        print('The simulated gene ontology graph has been successfully generated.')

        return go_adj_norm

    hvg_mask = adata.var['highly_variable']
    hvg_indices = adata.var_names[hvg_mask].tolist()
    hvg_symbols = adata.var['gene_symbol'][hvg_mask].tolist()
    N = len(hvg_indices)

    if os.path.exists(graph_path):
        for file_name in os.listdir(graph_path):
            if not file_name.endswith('.npz'):
                continue
            file_path = os.path.join(graph_path, file_name)
            try:
                loaded = np.load(file_path, allow_pickle=True)
                saved_symbols = loaded['symbols'].tolist()
                saved_n_neighbors = loaded['n_neighbors'].item()
                if saved_symbols == hvg_symbols and saved_n_neighbors == n_neighbors:
                    print(f"Loaded pre-computed real gene ontology graph from '{file_path}'.")
                    go_adj_norm = sp.csr_matrix((loaded['data'], loaded['indices'], loaded['indptr']), shape=loaded['shape'])
                    return go_adj_norm
            except Exception:
                continue

    symbol2indices = {}
    for idx, sym in zip(hvg_indices, hvg_symbols):
        symbol2indices.setdefault(sym, []).append(idx)

    mg = mygene.MyGeneInfo()
    queries = mg.querymany(list(symbol2indices.keys()), scopes=['symbol'],
                           fields=['go'], species='human', returnall=False, verbose=False)

    index2go = {}
    for q in queries:
        sym = q.get('query')
        gos = set()
        for d in ['BP', 'MF', 'CC']:
            terms = q.get('go', {}).get(d, [])
            terms = [terms] if isinstance(terms, dict) else terms
            gos.update(t['id'] for t in terms)
        if gos and sym in symbol2indices:
            for idx in symbol2indices[sym]:
                index2go[idx] = gos

    godag = GODag(obo_path, prt=None)
    for idx, terms in index2go.items():
        index2go[idx] = {t for t in terms if t in godag}

    unique_gos = sorted(list(set.union(*index2go.values()) if index2go else set()))
    if not unique_gos:
        raise ValueError('Error: No valid gene ontology terms found.')

    K = len(unique_gos)
    go2idx = {go: i for i, go in enumerate(unique_gos)}
    go_ancs = {go: godag[go].get_all_parents() | {go} for go in unique_gos}
    go_ns = {go: godag[go].namespace for go in unique_gos}

    S_go = np.zeros((K, K), dtype=np.float32)
    for i in range(K):
        go_i = unique_gos[i]
        for j in range(i, K):
            go_j = unique_gos[j]
            if go_ns[go_i] == go_ns[go_j]:
                intersection_len = len(go_ancs[go_i] & go_ancs[go_j])
                union_len = len(go_ancs[go_i] | go_ancs[go_j])
                sim = intersection_len / union_len if union_len > 0 else 0.0
                S_go[i, j] = S_go[j, i] = sim

    T = np.zeros((N, K), dtype=np.float32)
    n_i = np.zeros(N, dtype=np.float32)
    for i, idx in enumerate(hvg_indices):
        if idx in index2go:
            for go in index2go[idx]:
                T[i, go2idx[go]] = 1.0
            n_i[i] = len(index2go[idx])

    n_i_inv = np.divide(1.0, n_i, out=np.zeros_like(n_i), where=n_i!=0)
    S_gene_raw = np.dot(np.dot(T, S_go), T.T) * n_i_inv[:, None] * n_i_inv[None, :]
    np.fill_diagonal(S_gene_raw, 0.0)

    row_idx = np.repeat(np.arange(N), n_neighbors)
    col_idx = np.argpartition(S_gene_raw, -n_neighbors, axis=1)[:, -n_neighbors:].flatten()
    weights = S_gene_raw[row_idx, col_idx]

    mask = weights > 0
    go_adj_raw = sp.csr_matrix((weights[mask], (row_idx[mask], col_idx[mask])), shape=(N, N))
    go_adj_raw = go_adj_raw.maximum(go_adj_raw.T)
    go_adj_norm = normalize_adj(go_adj_raw, True).astype(np.float32)

    print('The real gene ontology graph has been successfully generated.')

    if save_graph:
        if not os.path.exists(graph_path):
            os.makedirs(graph_path)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(graph_path, f'go_graph_{timestamp}.npz')
        np.savez_compressed(
            save_path, 
            data=go_adj_norm.data, 
            indices=go_adj_norm.indices, 
            indptr=go_adj_norm.indptr, 
            shape=go_adj_norm.shape, 
            symbols=hvg_symbols,
            n_neighbors=n_neighbors
        )
        print(f'The real gene ontology graph successfully saved to {save_path}.')

    return go_adj_norm


def build_tf_graph(adata, n_neighbors=20, dataset_type='real', 
                   fa_path='../data/feature_prior/mm10.fa',
                   graph_path='../data/feature_prior/tf_graphs/', save_graph=True):
    if dataset_type == 'sim':
        N = adata.shape[1]
        rows, cols, data = [], [], []

        for i in range(N):
            choices = np.concatenate([np.arange(i), np.arange(i + 1, N)])
            selected_cols = np.random.choice(choices, n_neighbors, replace=False)

            rows.extend([i] * n_neighbors)
            cols.extend(selected_cols)
            data.extend([1e-8] * n_neighbors)

        rows.extend(range(N))
        cols.extend(range(N))
        data.extend([1.0] * N)

        tf_adj_raw = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
        tf_adj_norm = normalize_adj(tf_adj_raw, False).astype(np.float32)

        print('The simulated transcription factor graph has been successfully generated.')

        return tf_adj_norm

    hvg_mask = adata.var['highly_variable']
    hvp_names = adata.var_names[hvg_mask].tolist()
    N = len(hvp_names)

    if os.path.exists(graph_path):
        for file_name in os.listdir(graph_path):
            if not file_name.endswith('.npz'):
                continue
            file_path = os.path.join(graph_path, file_name)
            try:
                loaded = np.load(file_path, allow_pickle=True)
                saved_peaks = loaded['peaks'].tolist()
                saved_n_neighbors = loaded['n_neighbors'].item()
                if saved_peaks == hvp_names and saved_n_neighbors == n_neighbors:
                    print(f"Loaded pre-computed real transcription factor graph from '{file_path}'.")
                    tf_adj_norm = sp.csr_matrix((loaded['data'], loaded['indices'], loaded['indptr']), shape=loaded['shape'])
                    return tf_adj_norm
            except Exception:
                continue

    sequences = []
    genome = Fasta(fa_path)

    for peak in hvp_names:
        chrom, coords = peak.split('-', 1)
        start, end = map(int, coords.split('-'))
        try:
            seq = str(genome[chrom][start:end]).upper()
        except KeyError:
            seq = 'N' * (end - start)
        sequences.append(seq)

    jdb = jaspardb(release='JASPAR2024')
    motifs = jdb.fetch_motifs(collection='CORE', tax_group=['vertebrates'])

    iupac_dict = {
        'A':'A', 'C':'C', 'G':'G', 'T':'T',
        'R':'[AG]', 'Y':'[CT]', 'S':'[GC]', 'W':'[AT]',
        'K':'[GT]', 'M':'[AC]', 'B':'[CGT]', 'D':'[AGT]',
        'H':'[ACT]', 'V':'[ACG]', 'N':'[ACGT]'
    }

    def consensus_to_regex(consensus):
        return ''.join(iupac_dict.get(c, c) for c in consensus.upper())

    regex_patterns = []
    for m in motifs:
        pattern = consensus_to_regex(m.consensus)
        regex_patterns.append(re.compile(pattern))

    K = len(regex_patterns)
    M = np.zeros((N, K), dtype=np.float32)

    for i, seq in enumerate(sequences):
        for j, pattern in enumerate(regex_patterns):
            M[i, j] = len(pattern.findall(seq))

    M_sum = M.sum(axis=1, keepdims=True)
    M = np.divide(M, M_sum, out=np.zeros_like(M), where=M_sum!=0)
    S_motif = cosine_similarity(M)
    np.fill_diagonal(S_motif, 0.0)

    row_idx = np.repeat(np.arange(N), n_neighbors)
    col_idx = np.argpartition(S_motif, -n_neighbors, axis=1)[:, -n_neighbors:].flatten()
    weights = S_motif[row_idx, col_idx]

    mask = weights > 0
    tf_adj_raw = sp.csr_matrix((weights[mask], (row_idx[mask], col_idx[mask])), shape=(N, N))
    tf_adj_raw = tf_adj_raw.maximum(tf_adj_raw.T)
    tf_adj_norm = normalize_adj(tf_adj_raw, True).astype(np.float32)

    print('The real transcription factor graph has been successfully generated.')

    if save_graph:
        if not os.path.exists(graph_path):
            os.makedirs(graph_path)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(graph_path, f'tf_graph_{timestamp}.npz')
        np.savez_compressed(
            save_path, 
            data=tf_adj_norm.data, 
            indices=tf_adj_norm.indices, 
            indptr=tf_adj_norm.indptr, 
            shape=tf_adj_norm.shape, 
            peaks=hvp_names, 
            n_neighbors=n_neighbors
        )
        print(f'The real transcription factor graph successfully saved to {save_path}.')

    return tf_adj_norm


def build_trans_graph(adata_rna, adata_pro, pro2gene_path='../data/feature_prior/pro2gene.csv', dataset_type='real'):
    if dataset_type == 'real':
        current_rna_names = adata_rna.var['gene_symbol'][adata_rna.var['highly_variable']].astype(str).tolist()
    else:
        current_rna_names = adata_rna.var['gene_symbol'].astype(str).tolist()
    current_pro_names = adata_pro.var['gene_symbol'].astype(str).tolist()

    if dataset_type == 'sim':
        trans_graph_np = np.ones((len(current_rna_names), len(current_pro_names)), dtype=np.float32)
        print('The simulated cross-modal translational graph has been successfully generated.')
        return trans_graph_np, current_rna_names, current_pro_names

    mapping_df = pd.read_csv(pro2gene_path)
    pro_col = mapping_df['protein name'].astype(str).values
    rna_col = mapping_df['gene name'].astype(str).values

    custom_mapping = {
        p.strip(): [
            g.strip() for g in g_raw.replace('/', ',').split(',') 
            if g.strip() and g.strip() != 'nan'
        ] 
        for p, g_raw in zip(pro_col, rna_col)
    }

    rna2idx = {name: i for i, name in enumerate(current_rna_names)}
    M_full_np = np.zeros((len(current_rna_names), len(current_pro_names)), dtype=np.float32)
    for pro_idx, pro_name in enumerate(current_pro_names):
        for target_gene in custom_mapping.get(pro_name, [pro_name]):
            if target_gene in rna2idx:
                M_full_np[rna2idx[target_gene], pro_idx] = 1.0

    valid_rna_indices = np.where(M_full_np.sum(axis=1) > 0)[0]
    valid_pro_indices = np.where(M_full_np.sum(axis=0) > 0)[0]

    trans_graph_np = M_full_np[valid_rna_indices][:, valid_pro_indices]
    valid_rna_names = [current_rna_names[i] for i in valid_rna_indices]
    valid_pro_names = [current_pro_names[i] for i in valid_pro_indices]

    print('The real cross-modal translational graph has been successfully generated.')

    return trans_graph_np, valid_rna_names, valid_pro_names


def build_reg_graph(adata_atac, adata_rna, peak2gene_path='../data/feature_prior/peak2gene.csv', dataset_type='real'):
    if dataset_type == 'real':
        current_atac_names = adata_atac.var_names[adata_atac.var['highly_variable']].astype(str).tolist()
        current_rna_names = adata_rna.var['gene_symbol'][adata_rna.var['highly_variable']].astype(str).tolist()
    else:
        current_atac_names = adata_atac.var_names.astype(str).tolist()
        current_rna_names = adata_rna.var['gene_symbol'].astype(str).tolist()

    if dataset_type == 'sim':
        reg_graph_np = np.ones((len(current_atac_names), len(current_rna_names)), dtype=np.float32)
        print('The simulated cross-modal regulatory graph has been successfully generated.')
        return reg_graph_np, current_atac_names, current_rna_names

    mapping_df = pd.read_csv(peak2gene_path)
    atac2idx = {name: i for i, name in enumerate(current_atac_names)}
    rna2idx = {name: i for i, name in enumerate(current_rna_names)}
    M_full_np = np.zeros((len(current_atac_names), len(current_rna_names)), dtype=np.float32)

    peak_col = mapping_df['peak name'].astype(str).values
    gene_col = mapping_df['gene name'].astype(str).values
    for p, g in zip(peak_col, gene_col):
        peak_name = p.strip()
        gene_name = g.strip()
        if peak_name in atac2idx and gene_name in rna2idx:
            M_full_np[atac2idx[peak_name], rna2idx[gene_name]] = 1.0

    valid_atac_indices = np.where(M_full_np.sum(axis=1) > 0)[0]
    valid_rna_indices = np.where(M_full_np.sum(axis=0) > 0)[0]

    reg_graph_np = M_full_np[valid_atac_indices][:, valid_rna_indices]
    valid_atac_names = [current_atac_names[i] for i in valid_atac_indices]
    valid_rna_names = [current_rna_names[i] for i in valid_rna_indices]

    print('The real cross-modal regulatory graph has been successfully generated.')

    return reg_graph_np, valid_atac_names, valid_rna_names