import numpy as np
import pandas as pd
import torch
import abc
from typing import Tuple, List

import ot
from sklearn.cluster import KMeans

class FeatureAligner(abc.ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.src_transition_matrix = None
        self.trg_transition_matrix = None
        self.coarse_codes_num = cfg.da_model.coarse_num_code
        self.save_output_path = cfg.save_output_path
        self.aligned_feat_idxs: List[int] = []
        self.non_aligned_feat_idxs: List[int] = []

    def set_src_trg_matrix(self, src_transition_matrix: torch.Tensor, trg_transition_matrix: torch.Tensor) -> None:
        self.src_transition_matrix = src_transition_matrix
        self.trg_transition_matrix = trg_transition_matrix

    def set_coarse_codebook(self, coarse_codebook: torch.Tensor) -> None:
        self.coarse_codebook = coarse_codebook

    def compute_metric(self):
        metrics_per_channel = self._compute_metric(self.src_transition_matrix, self.trg_transition_matrix)
        self._save_metrics_per_channel(metrics_per_channel)
        return metrics_per_channel

    @abc.abstractmethod
    def _compute_metric(self, src_transition_matrix: torch.Tensor, trg_transition_matrix: torch.Tensor) -> torch.Tensor:
        pass

    def _save_metrics_per_channel(self, metrics_per_channel: torch.Tensor) -> None:
        metrics_per_channel_pd = pd.DataFrame(metrics_per_channel.numpy(), columns=["distance_metric"])
        metrics_per_channel_pd.reset_index(inplace=True)
        metrics_per_channel_pd.columns = ["channel", "distance_metric"]
        metrics_per_channel_pd.sort_values(by="distance_metric", inplace=True, ascending=False)
        # print 
        for i in range(len(metrics_per_channel_pd)):
            print(f"Channel {metrics_per_channel_pd.iloc[i]['channel']} Distance: {metrics_per_channel_pd.iloc[i]['distance_metric']}")
        # sort by metric
        
        metrics_per_channel_pd['distance_rank'] = range(1, len(metrics_per_channel_pd) + 1)
        metrics_per_channel_pd['exp_num'] = self.cfg.exp_num
        # metrics_per_channel_pd['aligned_feat_idxs'] = str(aligned_feat_idxs.tolist())
        # metrics_per_channel_pd['len_aligned_feat_idxs'] = len(aligned_feat_idxs)
        # metrics_per_channel_pd['non_aligned_feat_idxs'] = str(non_aligned_feat_idxs.tolist()) 
        # metrics_per_channel_pd['len_non_aligned_feat_idxs'] = len(non_aligned_feat_idxs)
        metrics_per_channel_pd.to_csv(f"{self.save_output_path}/distance_metrics.csv", index=False)

class FrobNAligner(FeatureAligner):
    """Frobenius Norm based alignment"""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.aligner_name = "frobn"
        self.aligner_metric = cfg.feature_alignment.aligner.metric
        assert self.cfg.feature_alignment.aligner.aligner_name == self.aligner_name, f"Expected aligner name 'frobn', got {cfg.feature_alignment.aligner_name}"


    def _compute_metric(self, src_transition_matrix: torch.Tensor, trg_transition_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute the Frobenius Norm between the source and target transition matrices for each channel.
        """
        if src_transition_matrix.shape != trg_transition_matrix.shape:
            raise ValueError(f"Shape mismatch: src {src_transition_matrix.shape} vs trg {trg_transition_matrix.shape}")
        if src_transition_matrix.shape[1] != self.cfg.da_model.coarse_num_code:
            raise ValueError(f"Expected {self.cfg.da_model.coarse_num_code} states, got {src_transition_matrix.shape[1]}")

        channel_dim, _, _ = src_transition_matrix.shape

        metrics_per_channel = torch.zeros(channel_dim)
        for c in range(channel_dim):
            metrics_per_channel[c] = torch.norm(src_transition_matrix[c] - trg_transition_matrix[c], p="fro")
        return metrics_per_channel





class POTAligner(FeatureAligner):
    """Optimal Transport based alignment"""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.aligner_name = "pot"
        self.aligner_metric = cfg.feature_alignment.aligner.metric
        assert self.cfg.feature_alignment.aligner.aligner_name == self.aligner_name, f"Expected aligner name 'pot', got {cfg.feature_alignment.aligner_name}"

    def _compute_metric(self, src_transition_matrix: torch.Tensor, trg_transition_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute the Optimal Transport (OT) between the source and target transition matrices for each channel.
        """
        if src_transition_matrix.shape != trg_transition_matrix.shape:
            raise ValueError(f"Shape mismatch: src {src_transition_matrix.shape} vs trg {trg_transition_matrix.shape}")
        cost_matrix = self._calculate_cost_matrix()
        channel_dim, code_num_dim, _ = src_transition_matrix.shape

        metrics_per_channel = torch.zeros(channel_dim)
        for c in range(channel_dim):
            channel_cost = []
            src_matrix = src_transition_matrix[c]
            trg_matrix = trg_transition_matrix[c]
            # compute optimal transport plan for each row (code)
            for i in range(code_num_dim): # 8 in our case
                src_row = src_matrix[i]
                trg_row = trg_matrix[i]
                if src_row.sum() == 0 or trg_row.sum() == 0:
                    channel_cost.append(0)
                    continue
                # compute optimal transport plan
                ot_plan = ot.emd(src_row.cpu().numpy(), trg_row.cpu().numpy(), cost_matrix, numItermax=10000)
                ot_cost = ot_plan * cost_matrix
                channel_cost.append(ot_cost.sum())
            metrics_per_channel[c] = torch.tensor(channel_cost).mean()
        return metrics_per_channel

    def _calculate_cost_matrix(self):
        coarse_codebook = self.coarse_codebook # (num_code, d_model). E.g., (8, 64)
        # calculate pairwise cosine similarity
        cosine_similarity = torch.nn.functional.cosine_similarity(coarse_codebook.unsqueeze(1), coarse_codebook.unsqueeze(0), dim=-1)
        # get 1 - cosine similarity to get the cost.
        # High cosine similarity means the vectors are similar, so the cost should be low.
        # Low cosine similarity means the vectors are dissimilar, so the cost should be high.
        cost_matrix = 1 - cosine_similarity
        
        # make diagonal elements zero. Cost is zero for transporting to itself. e.g., code 0 in src to code 0 in trg.
        cost_matrix.fill_diagonal_(0)
        print(f"Cost matrix: {cost_matrix}") 

        return cost_matrix.cpu().numpy()
     