

from omegaconf import DictConfig
import torch
import pandas as pd
import numpy as np
import math
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

def import_src_da_lightmodule(cfg: DictConfig, log):
    """
    Load the data module based on the data name
    :param cfg: Configuration dictionary. [omegaconf.dictconfig.DictConfig]
    :return: Data module. [pytorch_lightning.LightningDataModule]
    """
    if cfg.light_model_src.light_name == "cls_light_myvqvae_double":
        from da_light_model.cls_light_train_src import LitModel
    elif cfg.light_model_src.light_name == "reg_light_myvqvae_double":
        from da_light_model.reg_light_train_src import LitModel
    else:
        raise ValueError(f"Unknown lightning model {cfg.light_model_src.light_name}. ")

    return LitModel(cfg)

def import_trg_da_lightmodule(cfg: DictConfig, src_model=None, pseudo_label=None, pseudo_label_index_map=None, kl_div_per_sample=None):
    """
    Load the data module based on the data name
    :param cfg: Configuration dictionary. [omegaconf.dictconfig.DictConfig]
    :return: Data module. [pytorch_lightning.LightningDataModule]
    """
    if cfg.light_model_trg.light_name == "cls_light_myvqvae_double_adapt":
        from da_light_model.cls_light_adapt_trg import LitModel
    elif cfg.light_model_trg.light_name == "reg_light_myvqvae_double_adapt":
        from da_light_model.reg_light_adapt_trg import LitModel
    else:
        raise ValueError(f"Unknown lightning model {cfg.light_model_trg.light_name}. ")

    return LitModel(cfg, src_model, pseudo_label, pseudo_label_index_map, kl_div_per_sample)

def save_cls_representations(cfg, predict_outputs, file_name):
    """
    Save the representations for the predicted outputs
    :param cfg: Configuration dictionary. [omegaconf.dictconfig.DictConfig]
    :param predict_outputs: Predicted outputs. [torch.tensor]
    :param file_name: File name to save the representations. [str]
    """
    y_true = torch.cat([x["y_true"] for x in predict_outputs]).cpu().numpy()
    cls_rep = torch.cat([x["cls_rep"] for x in predict_outputs]).cpu().numpy()

    # save as numpy
    np.save(f"{cfg.save_output_path}/{file_name}_y_true.npy", y_true)
    np.save(f"{cfg.save_output_path}/{file_name}_cls_rep.npy", cls_rep)
    print(f"Representations saved to {cfg.save_output_path}/{file_name}_y_true.npy and {cfg.save_output_path}/{file_name}_cls_rep.npy")




class MetricCalculator:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def calculate_reg_predict_outputs(self, predict_outputs):
        """
        Calculate the metrics for the predicted outputs
        :param predict_outputs: Predicted outputs. [torch.tensor]
        """
        y_true = torch.cat([x["y_true"] for x in predict_outputs]).cpu()
        y_true_actual = torch.cat([x["y_true_actual"] for x in predict_outputs]).cpu()
        y_pred = torch.cat([x["reg_pred"] for x in predict_outputs]).squeeze().cpu()
        denorm_factor = torch.cat([x["denorm_factor"] for x in predict_outputs]).cpu()
        y_pred_denorm = y_pred * denorm_factor

        rmse = torch.nn.functional.mse_loss(y_true_actual, y_pred_denorm, reduction='mean').sqrt()
        mse = torch.nn.functional.mse_loss(y_true_actual, y_pred_denorm, reduction='mean')
        mae = torch.nn.functional.l1_loss(y_true_actual, y_pred_denorm)
        print(f"rmse: {rmse:.3f} MAE: {mae:.3f}")

        if self.cfg.data.data_name == "cmapss":
            error = y_true_actual - y_pred_denorm
            score_rul = scoring_func(error.numpy())
            print(f"Score: {score_rul:.3f}")

        # make into dataframe
        pred_df = pd.DataFrame(
            {
                "y_true": y_true.numpy(),
                "y_pred": y_pred.numpy(),
                "y_true_actual": y_true_actual.numpy(),
                "y_pred_denorm": y_pred_denorm.numpy(),
            }
        )
        metric_df = pred_df
        metric_df['rmse'] = rmse.item()
        metric_df['score'] = score_rul
        metric_df['mse'] = mse.item()
        metric_df['mae'] = mae.item()
        return metric_df
    

    def calculate_cls_predict_outputs(self, predict_outputs):
        """
        Calculate the metrics for the predicted outputs
        :param predict_outputs: Predicted outputs. [torch.tensor]
        """
        y_true = torch.cat([x["y_true"] for x in predict_outputs]).cpu()
        y_pred_logits = torch.cat([x["cls_pred"] for x in predict_outputs]).cpu()
        y_pred_proba = torch.nn.functional.softmax(y_pred_logits, dim=-1)
        y_pred = torch.argmax(y_pred_proba, dim=-1).numpy()
        accuracy = accuracy_score(y_true, y_pred)

        if self.cfg.data.num_class == 2:
            auroc = roc_auc_score(y_true, y_pred_proba[:, 1])
            auprc = average_precision_score(y_true, y_pred_proba[:, 1])
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
        else:
            auroc = roc_auc_score(y_true, y_pred_proba, multi_class="ovo")
            auprc = average_precision_score(y_true, y_pred_proba)
            f1 = f1_score(y_true, y_pred, average="macro")
            precision = precision_score(y_true, y_pred, average="macro")
            recall = recall_score(y_true, y_pred, average="macro")
        print(f"Accuracy: {accuracy:.3f} AUROC: {auroc:.3f} AUPRC: {auprc:.3f} F1: {f1:.3f} Precision: {precision:.3f} Recall: {recall:.3f}")

        # make into dataframe
        pred_df = pd.DataFrame(
            {
                "y_true": y_true.numpy(),
                "y_pred": y_pred,
            }
        )
        logit_df = pd.DataFrame(y_pred_logits.numpy(),
                                columns=[f"cls_{i}" for i in range(self.cfg.data.num_class)])
        metric_df = pd.concat([logit_df, pred_df], axis=1)
        metric_df['accuracy'] = accuracy
        metric_df['auroc'] = auroc
        metric_df['auprc'] = auprc
        metric_df['f1'] = f1
        metric_df['precision'] = precision
        metric_df['recall'] = recall
        return metric_df
        


    def save_metrics(self, predict_outputs, file_name):
        if self.cfg.data.data_type == "cls":
            metric_df = self.calculate_cls_predict_outputs(predict_outputs)
        elif self.cfg.data.data_type == "reg":
            metric_df = self.calculate_reg_predict_outputs(predict_outputs)
        else: 
            raise ValueError(f"Unknown data type {self.cfg.data.data_type}. ")
        metric_df['exp_num'] = self.cfg.exp_num
        metric_df['seed'] = self.cfg.seed
        metric_df['metric'] = self.cfg.feature_alignment.aligner.metric
        metric_df.to_csv(f"{self.cfg.save_output_path}/{file_name}.csv", index=False)
        print(f"Metrics saved to {self.cfg.save_output_path}/{file_name}.csv")
        
def scoring_func(error_arr): 
    pos_error_arr = error_arr[error_arr >= 0] 
    neg_error_arr = error_arr[error_arr < 0]
    score = 0 
    for error in neg_error_arr:
            score = math.exp(-(error / 13)) - 1 + score 
    for error in pos_error_arr: 
            score = math.exp(error / 10) - 1 + score
    return score

def print_ascii_art(text):
    ascii_art = {
        'A': ['   _   ', '  / \\  ', ' / _ \\ ', '/ ___ \\', '/_/  \\_\\'],
        'D': [' ____  ', '|  _ \\ ', '| | | |', '| |_| |', '|____/ '],
        'E': [' _____ ', '| ____|', '|  _|  ', '| |___ ', '|_____|'],
        'I': [' _____ ', '|_   _|', '  | |  ', '  | |  ', ' |___|'],
        'M': [' __  __ ', '|  \\/  |', '| |\\/| |', '| |  | |', '|_|  |_|'],
        'N': [' _   _ ', '| \\ | |', '|  \\| |', '| |\\  |', '|_| \\_|'],
        'O': [' _____ ', '|  _  |', '| | | |', '| |_| |', '|_____|'],
        'P': [' ____  ', '|  _ \\ ', '| |_) |', '|  __/ ', '|_|    '],
        'T': [' _____ ', '|_   _|', '  | |  ', '  | |  ', '  |_|  '],
        ' ': ['     ', '     ', '     ', '     ', '     ']
    }
    
    for i in range(5):
        for char in text.upper():
            print(ascii_art.get(char, [' ' * 7]*5)[i], end='')
        print()


    
