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

        self.linear_dropout = myvqvae_cfg.linear_dropout
        self.num_class = cfg.data.num_class

        self.kd_loss = KDLoss(cfg)
        self.kd_lambda = self.cfg.light_model_trg.kd.kd_lambda
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

        self.linear_dropout = nn.Dropout(p=self.linear_dropout)
        self.counter = 0

    
    def forward(self, x, src_mask=None, y_true=None, pseudo_label=None):
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

        loss, ce_loss, kd_loss, accuracy = 0, 0, 0, 0
        cls_pred = None
        if y_true is not None:
            # CLS token prediction
            cls_avg = reduce(cls_rep, "B C d_model -> B d_model", "mean")
            cls_pred = self.cls_prediction_head(cls_avg)
            cls_pred_src = cls_pred[src_mask]
            y_true = y_true[src_mask]
            ce_loss = F.cross_entropy(cls_pred_src, y_true)
            # calculate accuracy 
            pred_cls_src = cls_pred_src.argmax(dim=1)
            correct = (pred_cls_src == y_true).sum().item()
            accuracy = correct / y_true.numel()

            # Target KL divergence loss 
            if pseudo_label is not None:
                cls_pred_trg = cls_pred[~src_mask]
                # kd_loss = self.kd_lambda * self.kd_loss(cls_pred_trg, pseudo_label)
                kd_loss = self.kd_loss(cls_pred_trg, pseudo_label)
            loss = self.awl(ce_loss, kd_loss)
            # loss = ce_loss + kd_loss

        
        return loss, {
            "ce_loss": ce_loss,
            "kd_loss": kd_loss,
            "accuracy": accuracy,
            "cls_pred": cls_pred,
        }

    @staticmethod
    def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError(f"activation should be relu/gelu, not {activation}")

