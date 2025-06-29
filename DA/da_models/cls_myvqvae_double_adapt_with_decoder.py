import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from typing import Callable
from src.vqvae_layers.Embed import PositionalEmbedding
from src.vqvae_layers.vqmtm_backbones import Trans_Conv
from src.vqvae_layers.myvqvae_double_backbones import CoarseQuantizer, FineQuantizer
from da_loss.distillation import KDLoss
from da_loss.awl import AutomaticWeightedLoss
from einops import rearrange, reduce
from typing import Optional


class MyVQVAEDoubleAdapt(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        myvqvae_cfg = cfg.da_model

        # Hyper-parameters
        self.patch_len = myvqvae_cfg.patch_len
        self.task_name = cfg.task.task_name
        self.hidden_channels = myvqvae_cfg.hidden_channels
        self.d_model = myvqvae_cfg.d_model
        self.activation = myvqvae_cfg.activation
        self.drop_out = myvqvae_cfg.drop_out
        self.in_channels = cfg.data.c_in
        self.num_layers = myvqvae_cfg.num_layers
        self.coarse_num_code = myvqvae_cfg.coarse_num_code
        self.fine_num_code = myvqvae_cfg.fine_num_code
        self.commitment_cost = myvqvae_cfg.commitment_cost
        self.linear_dropout = myvqvae_cfg.linear_dropout
        self.num_class = cfg.data.num_class

        self.kd_loss = KDLoss(cfg)
        # instance norm
        self.instance_norm = nn.InstanceNorm1d(self.patch_len)

        # Project the patch_dim to the d_model dimension
        self.embed = nn.Linear(self.patch_len, self.d_model)
        self.activation_embed = self._get_activation_fn(self.activation)
        self.patch_norm = nn.LayerNorm(self.d_model)
        self.awl = AutomaticWeightedLoss(num=2)

        # embed the time series to patch
        self.pos_embed = PositionalEmbedding(
            d_model=self.d_model, max_len=100  # Number of patches.
        )

        # Transformer Encoder
        self.TOKEN_CLS = torch.normal(
            mean=0, std=0.02, size=(1, 1, 1, self.d_model)
        ).cuda()
        self.TOKEN_CLS.requires_grad = True
        self.register_buffer("CLS", self.TOKEN_CLS)

        self.encoder = nn.ModuleList(
            [
                Trans_Conv(
                    d_model=self.d_model,
                    dropout=self.drop_out,
                    in_channels=self.in_channels,
                    out_channels=self.in_channels,
                    activation=self.activation,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.cls_prediction_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.num_class),
        )

        self.coarse_vq = CoarseQuantizer(
            cfg=cfg,
            patch_len=self.patch_len,
            code_dim=self.d_model,
            num_code=self.coarse_num_code,
            commitment_cost=self.commitment_cost,
            kmeans_init=False,
        )

        self.fine_vq = FineQuantizer(
            cfg=cfg,
            patch_len=self.patch_len,
            code_dim=self.d_model,
            num_code=self.fine_num_code,
            commitment_cost=self.commitment_cost,
            kmeans_init=False,
        )

        self.activation = self._get_activation_fn(self.activation)
        self.decoder = nn.ModuleList(
            [
                Trans_Conv(
                    d_model=self.d_model,
                    dropout=self.drop_out,
                    in_channels=self.in_channels,
                    out_channels=self.in_channels,
                    activation=self.activation,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.reconstruct_decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Tanh(),
            nn.Linear(self.d_model, self.patch_len),
        )
        self.linear_dropout = nn.Dropout(p=self.linear_dropout)
        self.counter = 0

    
    def forward(self, x, src_mask=None, y_true=None, pseudo_label=None, kl_div_per_sample=None):
        # x: (B, Channel, Time)
        B, C, T = x.shape
        assert (
            T % self.patch_len == 0
        ), f"Time series length should be divisible by patch_len, not {T} % {self.patch_len}"
        x = x.view(B, C, -1, self.patch_len)  # (B, C, T, patch_len)

        # Embedding
        encoded_x = self.embed(x)  # (B, C, T, d_model)
        encoded_x = self.activation_embed(encoded_x)
        encoded_x = self.patch_norm(encoded_x)  # (B, C, T, d_model)

        # Add CLS token
        encoded_x = torch.concat(
            [self.CLS.repeat(B, C, 1, 1), encoded_x], dim=2
        )  # (B, C, T+1, d_model)
        encoded_x = encoded_x.view(-1, *encoded_x.shape[2:])
        pos_embed = self.pos_embed(encoded_x)  # (B*C, T+1, d_model)
        encoded_x = encoded_x + pos_embed

        # Transformer Encoder
        encoded_x = encoded_x.view(B, -1, *encoded_x.shape[1:])  # (B, C', T+1, d_model)
        for encoder in self.encoder:
            encoded_x = encoder(encoded_x)  # (B, C, T+1, d_model)
        cls_rep = encoded_x[:, :, 0, :]  # (B, C, d_model)
        encoded_patch_x = encoded_x[:, :, 1:, :]  # (B, C, T, d_model)

        (
            coarse_vq_loss,
            coarse_quantized_x,
            coarse_perplexity,
            coarse_embedding_weight,
            coarse_encoding_indices,
        ) = self.coarse_vq(encoded_patch_x)

        diff_patch_x = encoded_patch_x - coarse_quantized_x

        (
            fine_vq_loss,
            fine_quantized_x,
            fine_perplexity,
            fine_embedding_weight,
            fine_encoding_indices,
        ) = self.fine_vq(diff_patch_x)

        vq_loss = coarse_vq_loss + fine_vq_loss
        quantized_x = coarse_quantized_x + fine_quantized_x

        # Decoder
        for decoder in self.decoder:
            quantized_x = decoder(quantized_x)

        quantized_x = self.reconstruct_decoder(quantized_x)  # (B, C, T, patch_len)
        data_recon = rearrange(
            quantized_x, "B C T patch_len -> B C (T patch_len)"
        )  # (B, C, T*patch_len)
        recon_loss = F.mse_loss(data_recon, x.view(B, C, -1))  # (B, C, T*patch_len)

        vqvae_loss = recon_loss + vq_loss

        loss, ce_loss, kd_loss, accuracy = 0, 0, 0, 0
        cls_pred = None
        if y_true is not None:
            # CLS token prediction
            cls_avg = reduce(cls_rep, "B C d_model -> B d_model", "mean")
            cls_pred = self.cls_prediction_head(cls_avg)
            cls_pred_src = cls_pred[src_mask]
            y_true = y_true[src_mask]
            # There are corner cases where src_mask is all False.
            if y_true.numel() != 0:
                ce_loss = F.cross_entropy(cls_pred_src, y_true)
                # calculate accuracy 
                pred_cls_src = cls_pred_src.argmax(dim=1)
                correct = (pred_cls_src == y_true).sum().item()
                accuracy = correct / y_true.numel()
            else:
                ce_loss = 0
                accuracy = 0

            # Target KL divergence loss 
            if pseudo_label is not None:
                cls_pred_trg = cls_pred[~src_mask]

                # Soft Label
                if self.cfg.da_model.pseudo_label_confidence_topk_sampling:
                    # Ver 1: apply ce for topK percent of confident samples and KL for the rest
                    # apply ce for topK percent of confident samples and KL for the rest
                    # pseudo_label_hard_label = pseudo_label.argmax(dim=1)
                    # # larger distance = far from uniform, and thus confident.
                    # _, confident_k_idx = torch.topk(kl_div_per_sample, int(len(kl_div_per_sample) * self.cfg.da_model.pseudo_topk_percent), largest=True)
                    # trg_ce_loss = F.cross_entropy(cls_pred_trg[confident_k_idx], pseudo_label_hard_label[confident_k_idx])
                    # kd_loss = self.kd_loss(cls_pred_trg[~confident_k_idx], pseudo_label[~confident_k_idx])
                    # ce_loss += trg_ce_loss

                    # Ver 1-1: apply ce for topK percent of confident samples and KL for the rest
                    # apply ce for topK percent of confident samples and KL for the rest
                    # pseudo_label_hard_label = pseudo_label.argmax(dim=1)
                    # # larger distance = far from uniform, and thus confident.
                    # _, confident_k_idx = torch.topk(kl_div_per_sample, int(len(kl_div_per_sample) * self.cfg.da_model.pseudo_topk_percent), largest=True)
                    # trg_ce_loss = F.cross_entropy(cls_pred_trg[confident_k_idx], pseudo_label_hard_label[confident_k_idx])
                    # kd_loss = self.kd_loss(cls_pred_trg, pseudo_label)
                    # ce_loss += trg_ce_loss

                    # Ver 2: apply ce for topK percent of confident samples and KL for the rest
                    # apply ce for topK percent of confident samples and KL for the rest
                    # pseudo_label_hard_label = pseudo_label.argmax(dim=1)
                    # # larger distance = far from uniform, and thus confident.
                    # _, confident_k_idx = torch.topk(kl_div_per_sample, int(len(kl_div_per_sample) * self.cfg.da_model.pseudo_topk_percent), largest=True)
                    # _, non_confident_k_idx = torch.topk(kl_div_per_sample, int(len(kl_div_per_sample) * self.cfg.da_model.pseudo_topk_percent), largest=False)
                    # trg_ce_loss = F.cross_entropy(cls_pred_trg[confident_k_idx], pseudo_label_hard_label[confident_k_idx])
                    # kd_loss = self.kd_loss(cls_pred_trg[~non_confident_k_idx], pseudo_label[~non_confident_k_idx])
                    # ce_loss += trg_ce_loss

                    # Ver 3: apply KL for topK percent of confident samples only
                    # _, confident_k_idx = torch.topk(kl_div_per_sample, int(len(kl_div_per_sample) * self.cfg.da_model.pseudo_topk_percent), largest=True)
                    # kd_loss = self.kd_loss(cls_pred_trg[confident_k_idx], pseudo_label[confident_k_idx])
                

                    # Ver 4: Give Hard Label for confident samples
                    pseudo_label_hard_label = pseudo_label.argmax(dim=1)
                    # larger distance = far from uniform, and thus confident.
                    _, confident_k_idx = torch.topk(kl_div_per_sample, max(int(len(kl_div_per_sample) * self.cfg.da_model.pseudo_topk_percent), 1), largest=True)
                    trg_ce_loss = F.cross_entropy(cls_pred_trg[confident_k_idx], pseudo_label_hard_label[confident_k_idx])
                    # kd_loss = self.kd_loss(cls_pred_trg[~confident_k_idx], pseudo_label[~confident_k_idx])
                    kd_loss = torch.tensor(0).to(cls_pred_trg.device)
                    ce_loss += trg_ce_loss
                else:
                    kd_loss = self.kd_loss(cls_pred_trg, pseudo_label)
            
            # loss = self.awl(ce_loss, kd_loss, vqvae_loss)
            loss = self.awl(ce_loss, vqvae_loss)


        
        return loss, {
            "ce_loss": ce_loss,
            "kd_loss": kd_loss,
            "vqvae_loss": vqvae_loss,
            "accuracy": accuracy,
            "cls_pred": cls_pred,
            "cls_rep": cls_rep,
        }

    @staticmethod
    def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError(f"activation should be relu/gelu, not {activation}")

