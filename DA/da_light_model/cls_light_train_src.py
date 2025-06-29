import pandas as pd
import numpy as np
import os

from collections import defaultdict
import time

import lightning as L
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from src.utils import bcolors
from da_models.cls_myvqvae_double import MyVQVAEDouble


class LitModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.light_model_src.optimizer.lr
        self.patch_num = self.cfg.data.seq_len % self.cfg.da_model.patch_len
        self.initialize_losses(cfg)
        self.select_model()

    def training_step(self, batch, batch_idx):
        feature, y_true = batch["feature"], batch['y_true']
        feature = rearrange(feature, "b t c -> b c t")

        loss, outputs = self.model(feature, y_true)
        train_vq_loss = outputs["vq_loss"]
        train_recon_loss = outputs["recon_loss"]
        train_ce_loss = outputs["ce_loss"]
        train_coarse_perplexity = outputs["coarse_perplexity"]
        train_fine_perplexity = outputs["fine_perplexity"]
        train_accuracy = outputs["accuracy"]

        self.log("loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", train_accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log_losses(
            loss,
            train_vq_loss,
            train_recon_loss,
            train_ce_loss,
            train_coarse_perplexity,
            train_fine_perplexity,
            mode="train",
        )
        return loss

    def validation_step(self, batch, batch_idx):
        feature, y_true = batch["feature"], batch['y_true']
        feature = rearrange(feature, "b t c -> b c t")

        val_loss, outputs = self.model(feature, y_true)
        val_vq_loss = outputs["vq_loss"]
        val_recon_loss = outputs["recon_loss"]
        val_ce_loss = outputs["ce_loss"]
        val_coarse_perplexity = outputs["coarse_perplexity"]
        val_fine_perplexity = outputs["fine_perplexity"]

        self.log_losses(
            val_loss,
            val_vq_loss,
            val_recon_loss,
            val_ce_loss,
            val_coarse_perplexity,
            val_fine_perplexity,
            mode="val",
        )
        return val_loss

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.test_output_list = defaultdict(list)
        return

    def test_step(self, batch, batch_idx):
        original_feature, y_true = batch["feature"], batch["y_true"]
        B, T, C = original_feature.shape
        feature = rearrange(original_feature, "b t c -> b c t")

        test_loss, outputs = self.model(feature, mode="test")
        test_vq_loss = outputs["vq_loss"]
        test_recon_loss = outputs["recon_loss"]
        test_recon = outputs["data_recon"]
        test_coarse_perplexity = outputs["coarse_perplexity"]
        test_fine_perplexity = outputs["fine_perplexity"]
        test_coarse_encoding_indices = outputs["coarse_encoding_indices"]
        test_fine_encoding_indices = outputs["fine_encoding_indices"]

        test_recon = rearrange(test_recon, "b c t -> b t c")

        self.test_output_list["y_true"].append(y_true)
        self.test_output_list["original_feature"].append(original_feature)
        self.test_output_list["reconstructed_feature"].append(test_recon)
        self.test_output_list["test_loss"].append(test_loss)
        self.test_output_list["test_coarse_perplexity"].append(test_coarse_perplexity)
        self.test_output_list["test_fine_perplexity"].append(test_fine_perplexity)
        self.test_output_list["coarse_encoding_indices"].append(
            test_coarse_encoding_indices
        )
        self.test_output_list["fine_encoding_indices"].append(
            test_fine_encoding_indices
        )

    def on_test_epoch_end(self):
        outputs = self.test_output_list
        test_loss = torch.stack(outputs["test_loss"]).mean().cpu()
        coarse_perplexity = torch.stack(outputs["test_coarse_perplexity"]).mean().cpu()
        fine_perplexity = torch.stack(outputs["test_fine_perplexity"]).mean().cpu()
        y_true = torch.cat(outputs["y_true"], dim=0).cpu()
        original_feature = torch.cat(outputs["original_feature"], dim=0).cpu()
        reconstructed_feature = torch.cat(outputs["reconstructed_feature"], dim=0).cpu()
        coarse_encoding_indices = torch.cat(
            outputs["coarse_encoding_indices"], dim=0
        ).cpu()
        fine_encoding_indices = torch.cat(outputs["fine_encoding_indices"], dim=0).cpu()

        decoded_coarse_codebook, decoded_fine_codebook = self.model.decode_codebook()

        reconstruct_mse = F.mse_loss(reconstructed_feature, original_feature)
        reconstruct_mae = F.l1_loss(reconstructed_feature, original_feature)

        data_to_save = {
            "y_true": y_true,
            "original_feature": original_feature,
            "reconstructed_feature": reconstructed_feature,
            "coarse_encoding_indices": coarse_encoding_indices,
            "fine_encoding_indices": fine_encoding_indices,
            "reconstruct_mse": reconstruct_mse,
            "reconstruct_mae": reconstruct_mae,
            "coarse_perplexity": coarse_perplexity,
            "fine_perplexity": fine_perplexity,
            "decoded_coarse_codebook": decoded_coarse_codebook.cpu(),
            "decoded_fine_codebook": decoded_fine_codebook.cpu(),
        }

        # Save the dictionary to a file
        torch.save(
            data_to_save,
            f"{self.cfg.save_output_path}/cv{self.cfg.data.validation_cv_num}_combined_outputs.pt",
        )
        save_model_weight_path = (
            f"{self.cfg.save_output_path}/cv{self.cfg.data.validation_cv_num}_model.pt"
        )
        self.save_model_weight(save_model_weight_path)

        meta_data = self.make_meta_data(
            self.cfg,
            additional_dict={
                "test_loss": test_loss.item(),
                "coarse_perplexity": coarse_perplexity.item(),
                "fine_perplexity": fine_perplexity.item(),
                "reconstruct_mse": reconstruct_mse.item(),
                "reconstruct_mae": reconstruct_mae.item(),
            },
        )
        meta_data.to_csv(
            f"{self.cfg.save_output_path}/cv{self.cfg.data.validation_cv_num}_meta.csv",
            index=False,
        )

        self.log("test_loss", test_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "test_coarse_perplexity",
            coarse_perplexity,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_fine_perplexity",
            fine_perplexity,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "reconstruct_mse",
            reconstruct_mse,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "reconstruct_mae",
            reconstruct_mae,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def predict_step(self, batch, batch_idx):
        gid, feature, y_true = batch['global_id'], batch["feature"], batch['y_true']
        feature = rearrange(feature, "b t c -> b c t")

        _, outputs = self.model(feature, y_true)

        coarse_encoding_indices = outputs['coarse_encoding_indices']
        cls_pred = outputs['cls_pred']
        cls_rep = outputs['cls_rep']

        return {"coarse_encoding_indices": coarse_encoding_indices,
                "cls_pred": cls_pred,
                "cls_rep": cls_rep,
                "y_true": y_true,
                "gid": gid}
    def on_fit_end(self):
        # save coarse codebook
        coarse_codebook = self.model.get_codebook()['coarse']
        coarse_codebook = coarse_codebook.cpu().detach().numpy()
        np.save(f"{self.cfg.save_output_path}/src_coarse_codebook.npy", coarse_codebook)
    def initialize_losses(self, cfg):
        """Initialize loss functions based on configuration."""
        self.loss = nn.MSELoss()

    def make_meta_data(self, cfg, additional_dict=None):
        """make metadata."""
        meta_data = {
            "exp_num": cfg.exp_num,
            "seed": cfg.seed,
            "patch_len": cfg.da_model.patch_len,
            "coarse_num_code": cfg.da_model.coarse_num_code,
            "fine_num_code": cfg.da_model.fine_num_code,
            "d_model": cfg.da_model.d_model,
        }
        # make it pandas dataframe
        meta_data = pd.DataFrame(meta_data, index=[0])
        # add additional information if exists
        if additional_dict is not None:
            for key, value in additional_dict.items():
                meta_data[key] = value

        return meta_data

    def save_model_weight(self, path=None):
        model_weightname = f"cv{self.cfg.data.validation_cv_num}_da_model.pt"
        encoder_ckpt = {"model_state_dict": self.model.state_dict()}
        if path is None:
            root_path = self.cfg.save_pt_path
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            path = os.path.join(root_path, model_weightname)
        torch.save(encoder_ckpt, path)
        print("=====================================")
        print(f"Best pretrain model epoch: {self.current_epoch} saved to {path}")
        print("=====================================")

    def log_losses(
        self,
        total_loss,
        vq_loss,
        recon_loss,
        ce_loss,
        coarse_perplexity,
        fine_perplexity,
        mode="train",
    ):
        """Log losses."""
        self.log(f"{mode}_loss", total_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_vq_loss", vq_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            f"{mode}_recon_loss", recon_loss, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(f"{mode}_ce_loss", ce_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            f"{mode}_coarse_perplexity",
            coarse_perplexity,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{mode}_fine_perplexity",
            fine_perplexity,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def select_model(self):
        self.model = MyVQVAEDouble(self.cfg)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
