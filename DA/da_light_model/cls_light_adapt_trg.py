import pandas as pd
import numpy as np
import os

from collections import defaultdict
import time
from functools import partial

import lightning as L
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from src.utils import bcolors
# from da_models.cls_myvqvae_double_adapt import MyVQVAEDoubleAdapt
from da_models.cls_myvqvae_double_adapt_with_decoder import MyVQVAEDoubleAdapt


class LitModel(L.LightningModule):
    def __init__(self, cfg, src_model=None, pseudo_label=None, pseudo_label_index_map=None, kl_div_per_sample=None):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.light_model_trg.optimizer.lr
        self.patch_num = self.cfg.data.seq_len % self.cfg.da_model.patch_len
        self.select_trg_model()
        self.transfer_model_weights(src_model, self.trg_model)
        self.pseudo_label = pseudo_label.to(f"cuda:{self.cfg.gpu_id}")
        self.pseudo_label_index_map = pseudo_label_index_map
        self.kl_div_per_sample = kl_div_per_sample.to(f"cuda:{self.cfg.gpu_id}")
        

    def training_step(self, batch, batch_idx):
        gid, feature, y_true, source_mask = batch["global_id"], batch["feature"], batch['y_true'], batch['source_mask']
        source_mask = source_mask.bool() # mask is 1 for source and 0 for target
        trg_mask = ~source_mask
        trg_gid = gid[trg_mask]
        # get corresponding value from pseudo_label_index_map
        trg_index = torch.tensor([self.pseudo_label_index_map[g.item()] for g in trg_gid]).to(self.device)
        trg_pseudo_label = self.pseudo_label[trg_index]
        trg_kl_div_per_sample = self.kl_div_per_sample[trg_index]
        feature = rearrange(feature, "b t c -> b c t")

        loss, outputs = self.trg_model(feature, 
                                       src_mask=source_mask, 
                                       y_true=y_true,
                                       pseudo_label = trg_pseudo_label,
                                        kl_div_per_sample = trg_kl_div_per_sample)
        train_ce_loss = outputs["ce_loss"]
        train_kd_loss = outputs["kd_loss"]
        train_accuracy = outputs["accuracy"]
        # uncomment the below 
        train_vqvae_loss = outputs["vqvae_loss"]

        self.log("loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_ce_loss", train_ce_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_kd_loss", train_kd_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", train_accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_vqvae_loss", train_vqvae_loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        feature, y_true, source_mask = batch["feature"], batch['y_true'], batch['source_mask']
        source_mask = source_mask.bool()
        feature = rearrange(feature, "b t c -> b c t")

        val_loss, outputs = self.trg_model(feature, 
                                            src_mask=source_mask, 
                                            y_true=y_true)
        val_ce_loss = outputs["ce_loss"]
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return val_loss

    def predict_step(self, batch, batch_idx):
        if batch_idx==0:
            print(f"Predict step: {batch_idx}, Printing trainable modules")
            self.print_module_grad_status(self.trg_model)
        gid, feature, y_true, source_mask = batch['global_id'], batch["feature"], batch['y_true'], batch['source_mask']
        feature = rearrange(feature, "b t c -> b c t")
        source_mask = source_mask.bool()
        # For predict step in light_adapt_trg, we need to revert the source_mask to calculate accuracy.
        trg_mask = ~source_mask
        _, outputs = self.trg_model(feature,
                                src_mask=trg_mask,
                                y_true=y_true,
                                pseudo_label=None)
        # print(self.trg_model.cls_prediction_head[0].weight)
        cls_pred = outputs['cls_pred']
        cls_rep = outputs['cls_rep']

        return {"cls_pred": cls_pred,
                "cls_rep": cls_rep,
                "y_true": y_true,
                "gid": gid}


    def print_module_grad_status(self, model):
        def print_status(name, requires_grad):
            if requires_grad:
                print(f"{bcolors.OKGREEN}{name}: requires_grad=True (Trainable){bcolors.ENDC}")
            else:
                print(f"{bcolors.OKCYAN}{name}: requires_grad=False (Frozen){bcolors.ENDC}")

        for name, module in model.named_modules():
            if isinstance(module, nn.ParameterList):
                for i, param in enumerate(module):
                    print_status(f"ParameterList: {name}[{i}]", param.requires_grad)
            elif isinstance(module, nn.Parameter):
                print_status(f"Parameter: {name}", module.requires_grad)
            elif isinstance(module, nn.Embedding):
                print_status(f"Embedding: {name}", module.weight.requires_grad)
            elif hasattr(module, 'weight') or hasattr(module, 'bias'):
                if hasattr(module, 'weight') and module.weight is not None:
                    print_status(f"Module: {name} (Weight)", module.weight.requires_grad)
                if hasattr(module, 'bias') and module.bias is not None:
                    print_status(f"Module: {name} (Bias)", module.bias.requires_grad)



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
        fine_perplexity_trg,
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
            f"{mode}_fine_perplexity_trg",
            fine_perplexity_trg,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def select_trg_model(self):
        if self.cfg.da_model.da_model_name == "cls_myvqvae_double":
             self.trg_model = MyVQVAEDoubleAdapt(self.cfg)
        else:
            raise NotImplementedError(
                f"Model {self.cfg.da_model.da_model_name} not implemented"
            )
        
    def transfer_model_weights(self, src_model, trg_model):
        src_state_dict = src_model.state_dict()
        trg_state_dict = trg_model.state_dict()

        new_state_dict = {}
        transferred_params = 0
        total_params = 0

        for name, param in trg_state_dict.items():
            # print(name)
            total_params += param.numel()
            if name in src_state_dict:
                new_state_dict[name] = src_state_dict[name]
                transferred_params += param.numel()
            else:
                new_state_dict[name] = param
                print(f"Parameter {name} not found in source model, as such not transferred")
        
        self.trg_model.load_state_dict(new_state_dict)

        print(f"Total parameters in target model: {total_params}")
        print(f"Parameters transferred from source model: {transferred_params}")
        print(f"Percentage of parameters transferred: {transferred_params/total_params*100:.2f}%")
        del src_state_dict, trg_state_dict, new_state_dict
    

    def freeze_modules(self):
        modules_2_freeze = self.cfg.light_model_trg.freeze_module
        print(f"Freezing modules: {modules_2_freeze}")

        if modules_2_freeze is not None:
            for name, param in self.trg_model.named_parameters():
                if any([module in name for module in modules_2_freeze]):
                    param.requires_grad = False
                    print(f"{bcolors.OKCYAN} Freezing {name}{bcolors.ENDC}")
                else:
                    print(f"{bcolors.OKGREEN} Not freezing {name}{bcolors.ENDC}")

        # count the number of params that are trainable
        total_params = 0
        trainable_params = 0
        for param in self.trg_model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"Total parameters in target model: {total_params}")
        print(f"Total trainable parameters in target model: {trainable_params}")
        print(f"Percentage of trainable parameters: {trainable_params/total_params*100:.2f}%")
        



    def configure_optimizers(self):
        def linear_warmup(num_warmup_steps):
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return 1.0
            return lr_lambda

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, 
                            weight_decay=self.cfg.light_model_trg.optimizer.decay)

        if self.cfg.light_model_trg.scheduler.warmup_step > 0:
            num_warmup_steps = self.cfg.light_model_trg.scheduler.warmup_step

            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=linear_warmup(num_warmup_steps)
                ),
                'name': 'learning_rate',
                'interval': 'step',
                'frequency': 1
            }

            return [optimizer], [lr_scheduler]
        else:
            return optimizer