import os
import pandas as pd
import numpy as np
import pickle
from typing import Optional
from tqdm import tqdm


import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L

from src.utils import bcolors

class BaseDataModule(L.LightningDataModule):
    """Base class for data modules."""
    def __init__(self, cfg, data_type):
        super().__init__()
        self.cfg = cfg
        self.data_type = data_type  # "src" or "trg"
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _get_dataloader(self, dataset, shuffle, drop_last):
        config = getattr(self.cfg, f"light_model_{self.data_type}")
        return DataLoader(
            dataset,
            batch_size=config.dataset.batch_size,
            num_workers=config.dataset.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def train_dataloader(self, drop_last=True):
        return self._get_dataloader(self.train_dataset, shuffle=True, drop_last=drop_last)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, shuffle=False, drop_last=False)

class SRCDataModule(BaseDataModule):
    """Load only the src data for the model training."""
    def __init__(self, cfg):
        super().__init__(cfg, "src")

    def setup(self, stage: Optional[str]):
        print("setup called")
        if stage == "fit":
            self.train_dataset = WISDMDataset(self.data_type, "train", self.cfg)
            self.val_dataset = WISDMDataset(self.data_type, "test", self.cfg)
        elif stage in ["test", "predict"]: # leave it for consistency. Although we are not using.
            self.test_dataset = WISDMDataset(self.data_type, "test", self.cfg)
        else:
            raise ValueError(f"Unknown stage {stage}")
        
class TRGDataModule(BaseDataModule):
    """Load only the trg data for the model training."""
    def __init__(self, cfg):
        super().__init__(cfg, "trg")

    def setup(self, stage: Optional[str]):
        print("setup called")
        if stage == "predict":
            self.train_dataset = WISDMDataset(self.data_type, "train", self.cfg)
            self.test_dataset = WISDMDataset(self.data_type, "test", self.cfg)
        else:
            raise ValueError(f"Unknown stage {stage}")
        
    def train_dataloader(self, drop_last=True):
        return self._get_dataloader(self.train_dataset, shuffle=False, drop_last=False)
    
class SRCTRGDataModule(BaseDataModule):
    """Load both the src, trg data for the model training."""
    def __init__(self, cfg):
        super().__init__(cfg, "trg")

    def setup(self, stage: Optional[str]):
        print("setup called")
        if stage == "fit":
            self.train_dataset_src = WISDMDataset("src", "train", self.cfg)
            self.train_dataset_trg = WISDMDataset("trg", "train", self.cfg)
            self.train_dataset = torch.utils.data.ConcatDataset([self.train_dataset_src, self.train_dataset_trg])
            self.val_dataset = WISDMDataset("src", "test", self.cfg)
        elif stage in ["test", "predict"]:
            self.test_dataset = WISDMDataset("trg", "test", self.cfg)
        else:
            raise ValueError(f"Unknown stage {stage}")

class WISDMDataset(Dataset):
    """Load Gilon sensor data and meta data using global_id value. index is mapped to global id through label_df"""

    def __init__(self, source, mode, cfg):
        print(bcolors.OKBLUE + bcolors.BOLD + f"{source} - {mode} Mode" + bcolors.ENDC + bcolors.ENDC)
        self.source = source
        self.mode = mode # train, test, val
        self.cfg = cfg 
        self.source_wisdm_data_name = self.cfg.data.da_data.src
        self.target_wisdm_data_name = self.cfg.data.da_data.trg

        assert self.source_wisdm_data_name <=35, "The source_wisdm_data_name should be less than 30"
        assert self.target_wisdm_data_name <=35, "The target_wisdm_data_name should be less than 30"
        assert self.source_wisdm_data_name != self.target_wisdm_data_name, "The source_wisdm_data_name and target_wisdm_data_name should not be same"

        self.da_data_folder_dir = cfg.data.da_data.folder_dir
        base_dir = f"src/data/{self.da_data_folder_dir}"
        print(bcolors.OKGREEN + f"Loading data from: {base_dir}" + bcolors.ENDC)

        if self.source == "src":
            self.label_df = pd.read_csv(f"{base_dir}/USER{str(self.source_wisdm_data_name).zfill(2)}_{mode}Y.csv")
            self.label_df['source'] = 1
            with open(f"{base_dir}/USER{str(self.source_wisdm_data_name).zfill(2)}_{mode}X.pkl", "rb") as f:
                self.sensor_array = pickle.load(f)
            print(f"Loaded {self.source_wisdm_data_name}_{mode}X.pkl and {self.source_wisdm_data_name}_{mode}Y.csv")

        elif self.source == "trg":
            if mode in ["train", "test"]:
                self.label_df = pd.read_csv(f"{base_dir}/USER{str(self.target_wisdm_data_name).zfill(2)}_{mode}Y.csv")
            else:
                raise ValueError(f"Unknown mode {mode}")
            
            self.label_df['source'] = 0
            with open(f"{base_dir}/USER{str(self.target_wisdm_data_name).zfill(2)}_{mode}X.pkl", "rb") as f:
                self.sensor_array = pickle.load(f)
            print(f"Loaded {self.target_wisdm_data_name}_{mode}X.pkl and {self.target_wisdm_data_name}_{mode}Y.csv")
        else :
            raise ValueError(f"Unknown source {source}")
        """If you want to perform any ablation on the datasets, please do it here. all the features will be based on the label_df"""
        self.label_df = self.label_df.reset_index(drop=True)


    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, index):
        # TODO: Check if the weighted sampling is working properly
        global_id = self.label_df.iloc[index].GLOBAL_ID
        label = self.label_df[self.label_df["GLOBAL_ID"] == global_id]
        source_mask = label.source.item() # 1 for src, 0 for trg
        feature = self.sensor_array[int(global_id)]
        y_true = torch.tensor(label.y_true.values[0], dtype=torch.long)

        return {
            "feature": torch.tensor(feature, dtype=torch.float32),
            "y_true": y_true,
            "global_id": torch.tensor(global_id, dtype=torch.long),
            "source_mask": torch.tensor(source_mask, dtype=torch.long),
        }


