import time
import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim import Adam
from torch.nn.parameter import Parameter
from torch.nn.utils import clip_grad_norm_
from torch.nn.modules.module import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .utils import scipy_to_torch_sparse
from .preprocess import (compute_pca, build_spot_graphs, build_go_graph, 
                         build_tf_graph, build_trans_graph, build_reg_graph)


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.gcn_weight = Parameter(torch.FloatTensor(input_dim, latent_dim))
        self.skip_weight = Parameter(torch.FloatTensor(input_dim, latent_dim))

        torch.nn.init.xavier_uniform_(self.gcn_weight)
        torch.nn.init.xavier_uniform_(self.skip_weight)

    def forward(self, precomputed_ax, raw_x):
        h = torch.mm(precomputed_ax, self.gcn_weight)
        skip = torch.mm(raw_x, self.skip_weight)

        return h + skip


class Decoder(nn.Module):
    def __init__(self, latent_dim, recon_input_dim, n_branches=2):
        super(Decoder, self).__init__()
        self.n_branches = n_branches
        self.weight = Parameter(torch.FloatTensor(latent_dim, recon_input_dim))

        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, latent_emb, spat_adj, feat_adj=None):
        spot_h = torch.spmm(spat_adj, latent_emb)

        if self.n_branches == 1:
            x_recon = torch.mm(spot_h, self.weight)
        elif self.n_branches == 2:
            w_prior_smoothed = torch.spmm(feat_adj, self.weight.t()).t()
            w_fused = self.weight + w_prior_smoothed
            x_recon = torch.mm(spot_h, w_fused)

        return x_recon


class MAPLE(Module):
    def __init__(
        self,
        adata_rna=None,
        adata_pro=None,
        adata_atac=None,
        img_emb=None,
        dataset_type='real',
        latent_dim=64,
        n_spat_neighbors=6,
        n_feat_neighbors=20,
        n_go_neighbors=5,
        n_tf_neighbors=20,
        gamma_trans=1e-1,
        gamma_reg=1,
        lr=1e-2,
        n_epochs=2000,
        max_patience=200,
        weight_decay=1e-3,
        go_save_graph=True,
        tf_save_graph=True,
        go_obo_path='../data/feature_prior/go-basic.obo',
        tf_fa_path='../data/feature_prior/mm10.fa',
        go_graph_path='../data/feature_prior/go_graphs/',
        tf_graph_path='../data/feature_prior/tf_graphs/',
        pro2gene_path='../data/feature_prior/pro2gene.csv',
        peak2gene_path='../data/feature_prior/peak2gene.csv',
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ):
        super(MAPLE, self).__init__()

        self.adata_rna = adata_rna
        self.adata_pro = adata_pro
        self.adata_atac = adata_atac
        self.img_emb = img_emb
        self.dataset_type = dataset_type
        self.latent_dim = latent_dim
        self.n_spat_neighbors = n_spat_neighbors
        self.n_feat_neighbors = n_feat_neighbors
        self.n_go_neighbors = n_go_neighbors
        self.n_tf_neighbors = n_tf_neighbors
        self.gamma_trans = gamma_trans
        self.gamma_reg = gamma_reg
        self.lr = lr
        self.n_epochs = n_epochs
        self.max_patience = max_patience
        self.weight_decay = weight_decay
        self.go_save_graph = go_save_graph
        self.tf_save_graph = tf_save_graph
        self.go_obo_path = go_obo_path
        self.tf_fa_path = tf_fa_path
        self.go_graph_path = go_graph_path
        self.tf_graph_path = tf_graph_path
        self.pro2gene_path = pro2gene_path
        self.peak2gene_path = peak2gene_path
        self.device = device

        self.adata_list = [self.adata_rna, self.adata_pro, self.adata_atac]
        self.n_not_none_adatas = sum(1 for adata in self.adata_list if adata is not None)

        if self.n_not_none_adatas == 0:
            raise ValueError('At least one of the molecular modality data must be exists!')

        if self.adata_rna is not None:
            if self.dataset_type == 'real':
                raw_X_rna = self.adata_rna[:, self.adata_rna.var['highly_variable']].X
            else:
                raw_X_rna = self.adata_rna.X
            raw_X_rna = raw_X_rna.toarray() if sp.issparse(raw_X_rna) else np.array(raw_X_rna)
            self.input_dim_rna = raw_X_rna.shape[1]
            self.input_data_rna = torch.FloatTensor(raw_X_rna).to(self.device)

            spat_adj_rna, feat_adj_rna, joint_adj_rna = build_spot_graphs(self.adata_rna, \
                self.n_spat_neighbors, self.n_feat_neighbors)
            self.spat_adj_rna = scipy_to_torch_sparse(spat_adj_rna).to(self.device)
            self.feat_adj_rna = scipy_to_torch_sparse(feat_adj_rna).to(self.device)
            self.joint_adj_rna = scipy_to_torch_sparse(joint_adj_rna).to(self.device)

            go_adj = build_go_graph(self.adata_rna, self.n_go_neighbors, self.dataset_type,
                                        self.go_obo_path, self.go_graph_path, self.go_save_graph)
            self.go_adj = scipy_to_torch_sparse(go_adj).to(self.device)

            self.precomputed_joint_ax_rna = torch.spmm(self.joint_adj_rna, self.input_data_rna)
            self.encoder_rna = Encoder(self.input_dim_rna, self.latent_dim).to(self.device)
            self.decoder_rna = Decoder(self.latent_dim, self.input_dim_rna, 2).to(self.device)

        if self.adata_pro is not None:
            raw_X_pro = self.adata_pro.X
            raw_X_pro = raw_X_pro.toarray() if sp.issparse(raw_X_pro) else np.array(raw_X_pro)
            self.input_dim_pro = raw_X_pro.shape[1]
            self.input_data_pro = torch.FloatTensor(raw_X_pro).to(self.device)

            spat_adj_pro, feat_adj_pro, joint_adj_pro = build_spot_graphs(self.adata_pro, \
                self.n_spat_neighbors, self.n_feat_neighbors)
            self.spat_adj_pro = scipy_to_torch_sparse(spat_adj_pro).to(self.device)
            self.feat_adj_pro = scipy_to_torch_sparse(feat_adj_pro).to(self.device)
            self.joint_adj_pro = scipy_to_torch_sparse(joint_adj_pro).to(self.device)

            self.precomputed_joint_ax_pro = torch.spmm(self.joint_adj_pro, self.input_data_pro)
            self.encoder_pro = Encoder(self.input_dim_pro, self.latent_dim).to(self.device)
            self.decoder_pro = Decoder(self.latent_dim, self.input_dim_pro, 1).to(self.device)

        if self.adata_atac is not None:
            if self.dataset_type == 'real':
                raw_X_atac = self.adata_atac.obsm['X_feat']
                raw_X_atac_hvp = self.adata_atac[:, self.adata_atac.var['highly_variable']].X
            else:
                raw_X_atac = self.adata_atac.obsm['X_feat']
                raw_X_atac_hvp = self.adata_atac.X
            raw_X_atac = raw_X_atac.toarray() if sp.issparse(raw_X_atac) else np.array(raw_X_atac)
            raw_X_atac_hvp = raw_X_atac_hvp.toarray() if sp.issparse(raw_X_atac_hvp) else np.array(raw_X_atac_hvp)
            self.input_dim_atac = raw_X_atac.shape[1]
            self.input_dim_atac_hvp = raw_X_atac_hvp.shape[1]
            self.input_data_atac = torch.FloatTensor(raw_X_atac).to(self.device)
            self.input_data_atac_hvp = torch.FloatTensor(raw_X_atac_hvp).to(self.device)

            spat_adj_atac, feat_adj_atac, joint_adj_atac = build_spot_graphs(self.adata_atac, \
                self.n_spat_neighbors, self.n_feat_neighbors)
            self.spat_adj_atac = scipy_to_torch_sparse(spat_adj_atac).to(self.device)
            self.feat_adj_atac = scipy_to_torch_sparse(feat_adj_atac).to(self.device)
            self.joint_adj_atac = scipy_to_torch_sparse(joint_adj_atac).to(self.device)

            tf_adj = build_tf_graph(self.adata_atac, self.n_tf_neighbors, self.dataset_type,
                                        self.tf_fa_path, self.tf_graph_path, self.tf_save_graph)
            self.tf_adj = scipy_to_torch_sparse(tf_adj).to(self.device)

            self.precomputed_joint_ax_atac = torch.spmm(self.joint_adj_atac, self.input_data_atac)
            self.encoder_atac = Encoder(self.input_dim_atac, self.latent_dim).to(self.device)
            self.decoder_atac = Decoder(self.latent_dim, self.input_dim_atac, 1).to(self.device)
            self.decoder_atac_hvp = Decoder(self.latent_dim, self.input_dim_atac_hvp, 2).to(self.device)

        if self.img_emb is not None:
            img_emb_pca = compute_pca(self.img_emb, self.latent_dim)
            self.latent_emb_img = torch.FloatTensor(img_emb_pca).to(self.device)

        if self.adata_rna is not None and self.adata_pro is not None:
            trans_graph, self.valid_rna_names_with_pro, self.valid_pro_names_with_rna = \
                build_trans_graph(self.adata_rna, self.adata_pro, self.pro2gene_path, self.dataset_type)
            self.trans_graph = torch.FloatTensor(trans_graph).to_sparse().to(self.device)
            self.trans_decoder = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, self.input_dim_pro)
            ).to(self.device)

        if self.adata_atac is not None and self.adata_rna is not None:
            reg_graph, self.valid_atac_names_with_rna, self.valid_rna_names_with_atac = \
                build_reg_graph(self.adata_atac, self.adata_rna, self.peak2gene_path, self.dataset_type)
            self.reg_graph = torch.FloatTensor(reg_graph).to_sparse().to(self.device)
            self.reg_decoder = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, self.input_dim_rna)
            ).to(self.device)

        self.opt = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.opt, factor=0.5, patience=50)

    def forward(self):
        active_embs, results = [], {}

        if self.adata_rna is not None:
            latent_emb_rna = self.encoder_rna(self.precomputed_joint_ax_rna, self.input_data_rna)
            recon_input_rna = self.decoder_rna(latent_emb_rna, self.spat_adj_rna, self.go_adj)
            active_embs.append(latent_emb_rna)
            results.update({'latent_emb_rna': latent_emb_rna, 'recon_input_rna': recon_input_rna})

        if self.adata_pro is not None:
            latent_emb_pro = self.encoder_pro(self.precomputed_joint_ax_pro, self.input_data_pro)
            recon_input_pro = self.decoder_pro(latent_emb_pro, self.spat_adj_pro)
            active_embs.append(latent_emb_pro)
            results.update({'latent_emb_pro': latent_emb_pro, 'recon_input_pro': recon_input_pro})

        if self.adata_atac is not None:
            latent_emb_atac = self.encoder_atac(self.precomputed_joint_ax_atac, self.input_data_atac)
            recon_input_atac = self.decoder_atac(latent_emb_atac, self.spat_adj_atac)
            recon_input_atac_hvp = self.decoder_atac_hvp(latent_emb_atac, self.spat_adj_atac, self.tf_adj)
            active_embs.append(latent_emb_atac)
            results.update({'latent_emb_atac': latent_emb_atac, 'recon_input_atac':
                            recon_input_atac, 'recon_input_atac_hvp': recon_input_atac_hvp})

        if self.img_emb is not None:
            active_embs.append(self.latent_emb_img)
            results['latent_emb_img'] = self.latent_emb_img

        if self.adata_rna is not None and self.adata_pro is not None:
            trans_pro = self.trans_decoder(latent_emb_rna)
            results['trans_pro'] = trans_pro

        if self.adata_atac is not None and self.adata_rna is not None:
            reg_rna = self.reg_decoder(latent_emb_atac)
            results['reg_rna'] = reg_rna

        if len(active_embs) > 1:
            embs_stack = torch.stack(active_embs, dim=1)
            joint_emb = torch.mean(embs_stack, dim=1)
            results['joint_emb'] = joint_emb

        return results

    def train_modal(self):
        self.train()

        patience = 0
        min_loss = float('inf')

        start_time = time.time()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

        self.loss_dict = {'total_loss': []}
        if self.adata_rna is not None:
            self.loss_dict['recon_loss_rna'] = []
        if self.adata_pro is not None:
            self.loss_dict['recon_loss_pro'] = []
        if self.adata_atac is not None:
            self.loss_dict['recon_loss_atac'] = []
            self.loss_dict['recon_loss_atac_hvp'] = []
        if self.adata_rna is not None and self.adata_pro is not None:
            self.loss_dict['trans_loss'] = []
        if self.adata_atac is not None and self.adata_rna is not None:
            self.loss_dict['reg_loss'] = []

        if self.adata_rna is not None and self.adata_pro is not None:
            self.valid_pro_mask = None
            if self.trans_graph.shape[1] == self.input_dim_pro:
                if self.trans_graph.is_sparse:
                    self.valid_pro_mask = torch.zeros(self.input_dim_pro, dtype=torch.bool, device=self.device)
                    self.valid_pro_mask[self.trans_graph.indices()[1]] = True
                else:
                    self.valid_pro_mask = (self.trans_graph.sum(dim=0) > 0)

        if self.adata_atac is not None and self.adata_rna is not None:
            self.valid_rna_mask = None
            if self.reg_graph.shape[1] == self.input_dim_rna:
                if self.reg_graph.is_sparse:
                    self.valid_rna_mask = torch.zeros(self.input_dim_rna, dtype=torch.bool, device=self.device)
                    self.valid_rna_mask[self.reg_graph.indices()[1]] = True
                else:
                    self.valid_rna_mask = (self.reg_graph.sum(dim=0) > 0)

        pbar = tqdm(range(self.n_epochs), desc='Training')
        for epoch in pbar:
            results = self.forward()
            total_loss = 0.0
            postfix_dict = {}

            if self.adata_rna is not None:
                recon_loss_rna = F.mse_loss(self.input_data_rna, results['recon_input_rna'])
                postfix_dict['recon_loss_rna'] = f'{recon_loss_rna.item():.6f}'
                total_loss = total_loss + recon_loss_rna
                self.loss_dict['recon_loss_rna'].append(recon_loss_rna.item())

            if self.adata_pro is not None:
                recon_loss_pro = F.mse_loss(self.input_data_pro, results['recon_input_pro'])
                postfix_dict['recon_loss_pro'] = f'{recon_loss_pro.item():.6f}'
                total_loss = total_loss + recon_loss_pro
                self.loss_dict['recon_loss_pro'].append(recon_loss_pro.item())

            if self.adata_atac is not None:
                recon_loss_atac = F.mse_loss(self.input_data_atac, results['recon_input_atac'])
                recon_loss_atac_hvp = F.mse_loss(self.input_data_atac_hvp, results['recon_input_atac_hvp'])
                postfix_dict['recon_loss_atac'] = f'{recon_loss_atac.item():.6f}'
                postfix_dict['recon_loss_atac_hvp'] = f'{recon_loss_atac_hvp.item():.6f}'
                total_loss = total_loss + recon_loss_atac + {'real': 1.0, 'sim': 1e-8}[self.dataset_type] * recon_loss_atac_hvp
                self.loss_dict['recon_loss_atac'].append(recon_loss_atac.item())
                self.loss_dict['recon_loss_atac_hvp'].append(recon_loss_atac_hvp.item())

            if self.adata_rna is not None and self.adata_pro is not None:
                if self.valid_pro_mask is not None:
                    trans_loss = F.mse_loss(results['trans_pro'][:, self.valid_pro_mask], self.input_data_pro[:, self.valid_pro_mask])
                else:
                    trans_loss = F.mse_loss(results['trans_pro'], self.input_data_pro)
                total_loss = total_loss + {'real': self.gamma_trans, 'sim': 1e-8}[self.dataset_type] * trans_loss
                postfix_dict['trans_loss'] = f'{trans_loss.item():.6f}'
                self.loss_dict['trans_loss'].append(trans_loss.item())

            if self.adata_atac is not None and self.adata_rna is not None:
                if self.valid_rna_mask is not None:
                    reg_loss = F.mse_loss(results['reg_rna'][:, self.valid_rna_mask], self.input_data_rna[:, self.valid_rna_mask])
                else:
                    reg_loss = F.mse_loss(results['reg_rna'], self.input_data_rna)
                total_loss = total_loss + {'real': self.gamma_reg, 'sim': 1e-8}[self.dataset_type] * reg_loss
                postfix_dict['reg_loss'] = f'{reg_loss.item():.6f}'
                self.loss_dict['reg_loss'].append(reg_loss.item())

            self.loss_dict['total_loss'].append(total_loss.item())
            self.opt.zero_grad(set_to_none=True)
            total_loss.backward()
            clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.opt.step()
            self.scheduler.step(total_loss)

            postfix_dict['total_loss'] = f'{total_loss.item():.6f}'
            pbar.set_postfix_str(', '.join([f'{k}={v}' for k, v in postfix_dict.items()]))

            if total_loss.item() < min_loss - 1e-8:
                min_loss = total_loss.item()
                patience = 0
            else:
                patience += 1
                if patience >= self.max_patience:
                    print(f'Early stopping triggered at epoch {epoch + 1}.')
                    break

        total_time = time.time() - start_time
        print(f'Total training time: {total_time:.2f} seconds.')

        if self.device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            print(f'Peak GPU memory usage: {peak_memory:.2f} MB.')