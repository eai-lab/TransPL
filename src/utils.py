import numpy as np
import pandas as pd
import glob
import omegaconf
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import NeptuneLogger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

import torch


def seed_everything(seed: int = 42):
    """
    Seed everything for reproducibility.
    :param seed: Random seed. [int]
    :return:
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def import_src_datamodule(cfg: DictConfig, log):
    """
    Load the data module based on the data name
    :param cfg: Configuration dictionary. [omegaconf.dictconfig.DictConfig]
    :return: Data module. [pytorch_lightning.LightningDataModule]
    """
    if cfg.data.data_name in ["ucihar"]:
        log.info(f"Setting up UCIHAR Dataset")
        from src.dataset.ucihar_dataset import SRCDataModule
    elif cfg.data.data_name in ["wisdm"]:
        log.info(f"Setting up WISDM Dataset")
        from src.dataset.wisdm_dataset import SRCDataModule
    elif cfg.data.data_name in ["hhar"]:
        log.info(f"Setting up HHAR Dataset")
        from src.dataset.hhar_dataset import SRCDataModule
    elif cfg.data.data_name in ["ptb"]:
        log.info(f"Setting up PTB Dataset")
        from src.dataset.ptb_dataset import SRCDataModule
    else:
        raise ValueError(f"Unknown data name {cfg.data.data_name}")

    return SRCDataModule(cfg)

def import_trg_datamodule(cfg: DictConfig, log):
    """
    Load the data module based on the data name
    :param cfg: Configuration dictionary. [omegaconf.dictconfig.DictConfig]
    :return: Data module. [pytorch_lightning.LightningDataModule]
    """
    if cfg.data.data_name in ["ucihar"]:
        log.info(f"Setting up UCIHAR Dataset")
        from src.dataset.ucihar_dataset import TRGDataModule
    elif cfg.data.data_name in ["wisdm"]:
        log.info(f"Setting up WISDM Dataset")
        from src.dataset.wisdm_dataset import TRGDataModule
    elif cfg.data.data_name in ["hhar"]:
        log.info(f"Setting up HHAR Dataset")
        from src.dataset.hhar_dataset import TRGDataModule
    elif cfg.data.data_name in ["ptb"]:
        log.info(f"Setting up PTB Dataset")
        from src.dataset.ptb_dataset import TRGDataModule
    
    else:
        raise ValueError(f"Unknown data name {cfg.data.data_name}")

    return TRGDataModule(cfg)

def import_srctrg_datamodule(cfg: DictConfig, log):
    """
    Load the data module based on the data name
    :param cfg: Configuration dictionary. [omegaconf.dictconfig.DictConfig]
    :return: Data module. [pytorch_lightning.LightningDataModule]
    """
    if cfg.data.data_name in ["ucihar"]:
        log.info(f"Setting up UCIHAR Dataset")
        from src.dataset.ucihar_dataset import SRCTRGDataModule
    elif cfg.data.data_name in ["wisdm"]:
        log.info(f"Setting up WISDM Dataset")
        from src.dataset.wisdm_dataset import SRCTRGDataModule
    elif cfg.data.data_name in ["hhar"]:
        log.info(f"Setting up HHAR Dataset")
        from src.dataset.hhar_dataset import SRCTRGDataModule
    elif cfg.data.data_name in ["ptb"]:
        log.info(f"Setting up PTB Dataset")
        from src.dataset.ptb_dataset import SRCTRGDataModule
    else:
        raise ValueError(f"Unknown data name {cfg.data.data_name}")

    return SRCTRGDataModule(cfg)

def setup_neptune_logger(cfg: DictConfig, tags: list = None):
    """
    Nettune AI loger configuration. Needs API key.
    :param cfg: Configuration dictionary. [omegaconf.dictconfig.DictConfig]
    :param tags: List of tags to log a particular run. [list]
    :return:
    """

    # setup logger
    neptune_logger = NeptuneLogger(
        api_key=cfg.logger.api_key,
        project=cfg.logger.project_name,
        mode=cfg.logger.mode,
    )

    # neptune_logger.experiment["parameters/model"] = cfg.model.model_name

    return neptune_logger


def print_options(opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ""
    message += "----------------- Options ---------------\n"
    for k, v in sorted(vars(opt).items()):
        comment = ""
        message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
    message += "----------------- End -------------------"
    print(message)

def fix_config(cfg: DictConfig):
    num_gpu = torch.cuda.device_count()
    if cfg.gpu_id >= num_gpu:
        print(f"{bcolors.HEADER}{bcolors.WARNING} The gpu_id [{cfg.gpu_id}] exceeds the total GPUs [{num_gpu}]{bcolors.ENDC}{bcolors.ENDC}")
        print(f"{bcolors.HEADER}{bcolors.WARNING} Replace the gpu_id [{cfg.gpu_id}] to [{cfg.gpu_id % num_gpu}]{bcolors.ENDC}{bcolors.ENDC}")
        
        setattr(cfg, 'gpu_id', cfg.gpu_id % num_gpu)
    else:
        print(f"{bcolors.HEADER}{bcolors.OKGREEN} The gpu_id [{cfg.gpu_id}] is within the total GPUs [{num_gpu}]{bcolors.ENDC}{bcolors.ENDC}")

def bcolor_prints(cfg, show_full_cfg=False):
    fix_config(cfg)
    print(f"{bcolors.HEADER}=====> {cfg.task.task_name.upper()} <===== setting {bcolors.ENDC}")
    print(f"{bcolors.HEADER}=====> SRC LightModel {cfg.light_model_src.light_name.upper()} light model {bcolors.ENDC}")
    print(f"{bcolors.HEADER}=====> SRC LightModel {cfg.light_model_trg.light_name.upper()} light model {bcolors.ENDC}")
    print(f"{bcolors.HEADER}=====> with {cfg.data.data_name} data {bcolors.ENDC}")
    if show_full_cfg:
        print(f"\n\n{bcolors.HEADER}<========== Full Configurations ==========>\n{OmegaConf.to_yaml(cfg)} \n<=========================================>{bcolors.ENDC}\n\n")

class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def check_arg_errors(cfg):
    device_count = torch.cuda.device_count()
    assert cfg.gpu_id < device_count, f"gpu_id should be less than the number of available GPUs ({device_count})"

    if cfg.data.data_name in ["ucihar", "wisdm", "hhar", "ptb"]:
        assert cfg.data.da_data.src != cfg.data.da_data.trg, f"Source and Target data cannot be the same"
    else:
        raise ValueError(f"Unknown data name {cfg.data.data_name}")
    


    
