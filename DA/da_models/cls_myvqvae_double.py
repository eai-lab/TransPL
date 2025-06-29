import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from typing import Callable
from src.vqvae_layers.Embed import PositionalEmbedding
from src.vqvae_layers.vqmtm_backbones import Trans_Conv
from src.vqvae_layers.myvqvae_double_backbones import CoarseQuantizer, FineQuantizer
from einops import rearrange, reduce
from typing import Optional


class MyVQVAEDouble(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        myvqvae_cfg = cfg.da_model

        # Hyper-parameters
        self.cfg = cfg
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
        self.coarse_kmeans_init = myvqvae_cfg.coarse_kmeans_init
        self.fine_kmeans_init = myvqvae_cfg.fine_kmeans_init

        # instance norm
        self.instance_norm = nn.InstanceNorm1d(self.patch_len)

        # Project the patch_dim to the d_model dimension
        self.embed = nn.Linear(self.patch_len, self.d_model)
        self.activation_embed = self._get_activation_fn(self.activation)
        self.patch_norm = nn.LayerNorm(self.d_model)
        patch_num = cfg.data.seq_len // self.patch_len
        assert patch_num < 101, "The number of patches should be less than 100"
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
            kmeans_init=self.coarse_kmeans_init,
        )

        self.fine_vq = FineQuantizer(
            cfg=cfg,
            patch_len=self.patch_len,
            code_dim=self.d_model,
            num_code=self.fine_num_code,
            commitment_cost=self.commitment_cost,
            kmeans_init=self.fine_kmeans_init,
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

    def get_codebook(self):
        # return self.vq.codebook.weight.data
        return {
            "coarse": self.coarse_vq.codebook.weight.data,
        }

    def decode_codebook(self):
        x  = self.coarse_vq.codebook.weight.data
        patch_num = self.cfg.data.seq_len // self.patch_len 
        x = x.view(1, 1, -1, self.d_model)
        # Apply each decoder layer sequentially
        for decoder_layer in self.decoder:
            x = decoder_layer(x)
        # Apply the final reconstruction
        x = self.reconstruct_decoder(x)
        return x
    
    def forward(self, x, y_true=None):
        # x: (B, Channel, Time)
        B, C, T = x.shape
        assert (
            T % self.patch_len == 0
        ), f"Time series length should be divisible by patch_len, not {T} % {self.patch_len}"
        x = x.view(B, C, -1, self.patch_len)  # (B, C, patch_num, patch_len)

        # Embedding
        encoded_x = self.embed(x)  # (B, C, patch_num, d_model)
        encoded_x = self.activation_embed(encoded_x)
        encoded_x = self.patch_norm(encoded_x)  # (B, C, patch_num, d_model)

        # Add CLS token
        encoded_x = torch.concat(
            [self.CLS.repeat(B, C, 1, 1), encoded_x], dim=2
        )  # (B, C, patch_num+1, d_model)

        encoded_x = encoded_x.view(-1, *encoded_x.shape[2:]) # (B*C, patch_num+1, d_model)

        pos_embed = self.pos_embed(encoded_x)  # (B*C, patch_num+1, d_model)

        encoded_x = encoded_x + pos_embed

        # Transformer Encoder
        encoded_x = encoded_x.view(B, -1, *encoded_x.shape[1:])  # (B, C', patch_num+1, d_model)
        for encoder in self.encoder:
            encoded_x = encoder(encoded_x)  # (B, C, patch_num+1, d_model)
        cls_rep = encoded_x[:, :, 0, :]  # (B, C, d_model)
        encoded_patch_x = encoded_x[:, :, 1:, :]  # (B, C, patch_num, d_model)


        ## VQ-VAE
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

        if self.counter % 100 == 0:
            print("Coarse VQ")
            self.count_and_print(coarse_encoding_indices)
            print("Fine VQ")
            self.count_and_print(fine_encoding_indices)

        # Decoder
        for decoder in self.decoder:
            quantized_x = decoder(quantized_x)

        quantized_x = self.reconstruct_decoder(quantized_x)  # (B, C, T, patch_len)
        data_recon = rearrange(
            quantized_x, "B C T patch_len -> B C (T patch_len)"
        )  # (B, C, T*patch_len)
        recon_loss = F.mse_loss(data_recon, x.view(B, C, -1))  # (B, C, T*patch_len)

        loss = recon_loss + vq_loss 
        self.counter += 1

        ce_loss, accuracy = 0, 0
        if y_true is not None:
            # CLS token prediction
            cls_avg = reduce(cls_rep, "B C d_model -> B d_model", "mean")
            cls_pred = self.cls_prediction_head(cls_avg)
            ce_loss = F.cross_entropy(cls_pred, y_true)
            # calculate accuracy 
            pred_cls = cls_pred.argmax(dim=1)
            correct = (pred_cls == y_true).sum().item()
            accuracy = correct / B
            loss += ce_loss

        
        return loss, {
            "vq_loss": vq_loss,
            "recon_loss": recon_loss,
            "ce_loss": ce_loss,
            "data_recon": data_recon,
            "coarse_perplexity": coarse_perplexity,
            "fine_perplexity": fine_perplexity,
            "coarse_embedding_weight": coarse_embedding_weight,
            "fine_embedding_weight": fine_embedding_weight,
            "coarse_encoding_indices": coarse_encoding_indices,
            "fine_encoding_indices": fine_encoding_indices,
            "quantized_x": quantized_x,
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

    def count_and_print(self, idx):
        # Own Implemented. Count the occurrences of each unique element in the tensor
        total_elements = idx.numel()
        unique_elements, counts = torch.unique(idx, return_counts=True)

        # Combine the unique elements and their counts into a list of tuples
        element_counts = list(zip(unique_elements.tolist(), counts.tolist()))

        # Sort the list of tuples by the count in ascending order
        sorted_element_counts = sorted(element_counts, key=lambda x: x[1])

        # Get the top 10 most frequent elements (or all if there are fewer than 10)
        top_ten = sorted_element_counts[-10:]

        # Print each element and its count
        for element, count in top_ten:
            print(
                f"Element: {element}, Count: {count}, Percentage: {count / total_elements * 100:.2f}%"
            )
