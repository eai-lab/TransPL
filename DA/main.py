import os
import sys
import numpy as np

import datetime
import pickle
import logging
import gc
import time
import glob
from typing import Tuple, Dict, Any

import hydra
import lightning as L
import torch
from omegaconf import DictConfig
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary, LearningRateMonitor

from da_utilities.utilities import (
    import_src_da_lightmodule, import_trg_da_lightmodule, MetricCalculator, 
    print_ascii_art, save_cls_representations
)
from da_utilities.transition_matrix import TransitionMatrix
from da_utilities.feature_aligner import FrobNAligner, POTAligner
from da_utilities.likelihood import LikelihoodEvaluator

from src.utils import (
    bcolors, setup_neptune_logger, import_src_datamodule,
    import_trg_datamodule, import_srctrg_datamodule, seed_everything,
    bcolor_prints, check_arg_errors
)

torch.set_printoptions(sci_mode=False)
log = logging.getLogger(__name__)

def setup_environment(cfg: DictConfig) -> None:
    check_arg_errors(cfg)
    bcolor_prints(cfg, show_full_cfg=True)
    seed_everything(cfg.seed)

def flush_checkpoint_path(cfg, checkpoint_path: str) -> None:
    # remove .ckpt files in the checkpoint path
    if os.path.exists(f"{cfg.save_output_path}/checkpoints"):
        for file in os.listdir(checkpoint_path):
            if file.endswith(".ckpt"):
                os.remove(os.path.join(checkpoint_path, file))
                print(f"{bcolors.WARNING}Removed {file}{bcolors.ENDC}")

def setup_callbacks(cfg: DictConfig, checkpoint_path: str, is_source: bool) -> list:
    """Checkpoints, Early Stopping, Summary Callbacks"""
    model_cfg = cfg.light_model_src if is_source else cfg.light_model_trg
    suffix = "src_" if is_source else "trg_"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor=model_cfg.callbacks.monitor,
        save_top_k=1,
        filename=f"{suffix}" + cfg.da_model.da_model_name + "_{epoch:02d}",
        mode=model_cfg.callbacks.monitor_mode,
    )
    early_stop_callback = EarlyStopping(
        monitor=model_cfg.callbacks.monitor,
        patience=model_cfg.callbacks.patience,
        verbose=True,
        mode=model_cfg.callbacks.monitor_mode,
        min_delta=model_cfg.callbacks.min_delta,
    )
    model_summary_callback = ModelSummary(max_depth=1)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    return [checkpoint_callback, early_stop_callback, model_summary_callback, lr_monitor]

def setup_trainer(cfg: DictConfig, callbacks: list, logger: Any, is_source: bool) -> L.Trainer:
    """Setup Lightning Trainer."""
    model_cfg = cfg.light_model_src if is_source else cfg.light_model_trg
    return L.Trainer(
        accelerator="gpu",
        devices=[cfg.gpu_id],
        deterministic=cfg.deterministic,
        check_val_every_n_epoch=model_cfg.callbacks.check_val_every_n_epoch,
        max_epochs=model_cfg.callbacks.max_epochs,
        max_steps=model_cfg.callbacks.max_steps,
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=cfg.task.fast_dev_run,
        limit_train_batches=cfg.task.limit_train_batches,
        limit_val_batches=cfg.task.limit_val_batches,
    )

def train_source_model(cfg: DictConfig, trainer: L.Trainer, dm_src: Any) -> Tuple[Any, str]:
    """Pre-Train the Source Model"""
    light_model = import_src_da_lightmodule(cfg, log)
    trainer.fit(light_model, dm_src)
    best_src_ckpt_path = trainer.checkpoint_callback.best_model_path
    print(best_src_ckpt_path)
    best_train_epoch = int(best_src_ckpt_path.split("=")[-1].split(".")[0])
    log.info(f"Best Train Epoch for SRC model: {best_train_epoch}")
    return light_model, best_src_ckpt_path

def load_source_model(cfg: DictConfig,  dm_src: Any) -> Tuple[Any, str]:
    """Load a pre-trained Source Model"""
    light_model = import_src_da_lightmodule(cfg, log)
    
    if cfg.task.source_model.model_path is None:
        raise ValueError("Source model path is not provided.")
    
    # find the best checkpoint file
    best_src_ckpt_path = f"{cfg.task.source_model.model_path}/EXP{cfg.task.source_model.exp_num}/checkpoints/"
    best_src_ckpt_path = glob.glob(best_src_ckpt_path + "src_*.ckpt")
    assert len(best_src_ckpt_path) == 1, f"Multiple checkpoint files found: {best_src_ckpt_path}"
    best_src_ckpt_path = best_src_ckpt_path[0]

    if not os.path.exists(best_src_ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {best_src_ckpt_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(best_src_ckpt_path)
    light_model.load_state_dict(checkpoint['state_dict'])
    
    log.info(f"Loaded Source Model from checkpoint: {best_src_ckpt_path}")
    
    # Extract the epoch number from the checkpoint path
    best_train_epoch = int(best_src_ckpt_path.split("=")[-1].split(".")[0])
    log.info(f"Best Train Epoch for SRC model: {best_train_epoch}")

    log.info(f"Setting up the SRC datamodule for prediction")
    dm_src.setup(stage="fit")
    dm_src.setup(stage="predict")

def calculate_transition_matrices(cfg: DictConfig, trainer: L.Trainer, light_model: Any, dm_src: Any, dm_trg: Any, best_src_ckpt_path: str) -> Tuple[TransitionMatrix, MetricCalculator]:
    """Calculate Transition Matrices for SRC and"""
    trans_matrix = TransitionMatrix(cfg)
    metrics_calculator = MetricCalculator(cfg)
    
    #### Source train class-wise transition matrix
    predict_src_outputs = trainer.predict(light_model, dm_src.train_dataloader(), ckpt_path=best_src_ckpt_path, return_predictions=True)
    trans_matrix.calculate_src_classwise_transition_matrix(predict_src_outputs)
    ####### Source train transition matrix
    trans_matrix.calculate_src_transition_matrix(predict_src_outputs)
    if cfg.save_cls_representations:
        save_cls_representations(cfg, predict_src_outputs, "src_train")

    src_train_coarse_outputs = torch.cat([output['coarse_encoding_indices'] for output in predict_src_outputs]).cpu().detach().numpy()
    src_train_y_true = torch.cat([output['y_true'] for output in predict_src_outputs]).cpu().detach().numpy() 
    src_train_cls_pred = torch.cat([output['cls_pred'] for output in predict_src_outputs]).cpu().detach().numpy() 
    src_train_global_id = torch.cat([output['gid'] for output in predict_src_outputs]).cpu().detach().numpy() 
    src_output_dict = {
        "src_train_coarse_outputs": src_train_coarse_outputs,
        "src_train_y_true": src_train_y_true,
        "src_train_cls_pred": src_train_cls_pred,
        "src_train_global_id": src_train_global_id
    }
    # save the src_output_dict
    with open(f"{cfg.save_output_path}/src_output_dict.pkl", "wb") as f:
        pickle.dump(src_output_dict, f)
        print(f"Saved {cfg.save_output_path}/src_output_dict.pkl")
    del predict_src_outputs
    gc.collect()

    #### Source test transition matrix -> This is used for interpretability evaluation. 
    predict_src_test_outputs = trainer.predict(light_model, dm_src.test_dataloader(), ckpt_path=best_src_ckpt_path, return_predictions=True)
    src_test_transition_matrix = trans_matrix.calculate_transition_matrix(predict_src_test_outputs)
    trans_matrix.save_src_test_transition_matrix(src_test_transition_matrix)
    metrics_calculator.save_metrics(predict_src_test_outputs, "src_test_results")

    src_test_coarse_outputs = torch.cat([output['coarse_encoding_indices'] for output in predict_src_test_outputs]).cpu().detach().numpy()
    src_test_y_true = torch.cat([output['y_true'] for output in predict_src_test_outputs]).cpu().detach().numpy() 
    src_test_cls_pred = torch.cat([output['cls_pred'] for output in predict_src_test_outputs]).cpu().detach().numpy() 
    src_test_global_id = torch.cat([output['gid'] for output in predict_src_test_outputs]).cpu().detach().numpy()
    src_test_output_dict = {
        "src_test_coarse_outputs": src_test_coarse_outputs,
        "src_test_y_true": src_test_y_true,
        "src_test_cls_pred": src_test_cls_pred,
        "src_test_global_id": src_test_global_id
    }
    # save the src_test_output_dict
    with open(f"{cfg.save_output_path}/src_test_output_dict.pkl", "wb") as f:
        pickle.dump(src_test_output_dict, f)
        print(f"Saved {cfg.save_output_path}/src_test_output_dict.pkl")
    del predict_src_test_outputs, src_test_transition_matrix
    gc.collect()

    #### Target train transition matrix -> This is used to directly compare with source train transition matrix
    dm_trg.setup(stage="predict")
    predict_trg_train_outputs = trainer.predict(light_model, dm_trg.train_dataloader(), ckpt_path=best_src_ckpt_path, return_predictions=True)
    trans_matrix.calculate_trg_transition_matrix(predict_trg_train_outputs)
    if cfg.save_cls_representations:
        save_cls_representations(cfg, predict_trg_train_outputs, "trg_train")

    # Comment the below line if you don't want to save the outputs
    trg_train_coarse_outputs = torch.cat([output['coarse_encoding_indices'] for output in predict_trg_train_outputs]).cpu().detach().numpy()
    trg_train_y_true = torch.cat([output['y_true'] for output in predict_trg_train_outputs]).cpu().detach().numpy() 
    trg_train_cls_pred = torch.cat([output['cls_pred'] for output in predict_trg_train_outputs]).cpu().detach().numpy() 
    trg_train_global_id = torch.cat([output['gid'] for output in predict_trg_train_outputs]).cpu().detach().numpy() 
    trg_output_dict = {
        "trg_train_coarse_outputs": trg_train_coarse_outputs,
        "trg_train_y_true": trg_train_y_true,
        "trg_train_cls_pred": trg_train_cls_pred,
        "trg_train_global_id": trg_train_global_id
    }
    # save the trg_output_dict
    with open(f"{cfg.save_output_path}/trg_output_dict.pkl", "wb") as f:
        pickle.dump(trg_output_dict, f)
        print(f"Saved {cfg.save_output_path}/trg_output_dict.pkl")
    
    #### Save all transition matrices
    trans_matrix.save_transition_matrices()

    #### Target test model performance -> This is used to find out the performance of the model on the target domain without any adaptation
    predict_trg_test_outputs = trainer.predict(light_model, dm_trg.test_dataloader(), ckpt_path=best_src_ckpt_path, return_predictions=True)
    metrics_calculator.save_metrics(predict_trg_test_outputs, "trg_test_initial_results")

    # # Comment the below line if you don't want to save the outputs
    trg_test_coarse_outputs = torch.cat([output['coarse_encoding_indices'] for output in predict_trg_test_outputs]).cpu().detach().numpy()
    np.save(f"{cfg.save_output_path}/trg_test_coarse_outputs.npy", trg_test_coarse_outputs)

    return trans_matrix, metrics_calculator, predict_trg_train_outputs

def calculate_distance_per_channels(cfg: DictConfig, trans_matrix: TransitionMatrix, light_model: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate Global Distance Per Channels, between Source and Target Transition Matrices"""
    src_transition_m, trg_transition_m = trans_matrix.get_src_transition_matrix(), trans_matrix.get_trg_transition_matrix()
    if cfg.feature_alignment.aligner.aligner_name == "frobn":
        aligner = FrobNAligner(cfg) 
    elif cfg.feature_alignment.aligner.aligner_name == "pot":
        aligner = POTAligner(cfg)
        coarse_codebook = light_model.model.get_codebook()["coarse"]
        aligner.set_coarse_codebook(coarse_codebook)
    else:
        raise ValueError(f"Unknown feature alignment method {cfg.feature_alignment.aligner.aligner_name}.")
    
    aligner.set_src_trg_matrix(src_transition_m, trg_transition_m)
    distance_per_channel = aligner.compute_metric()
    return distance_per_channel

def get_pseudo_label_results(cfg, trans_matrix, predict_trg_train_outputs, distance_per_channel=None):
    """Construct Pseudo-Labels for Target Domain"""
    evaluator = LikelihoodEvaluator(cfg, trans_matrix, distance_per_channel)
    evaluator.construct_pseudo_label(predict_trg_train_outputs)
    trg_pseudo_label, trg_pseudo_label_index_map, kl_div_per_sample = evaluator.get_pseudo_label()
    print("Constructed Pseudo-Label")
    return trg_pseudo_label, trg_pseudo_label_index_map, kl_div_per_sample

def perform_domain_adaptation(cfg: DictConfig, light_model: Any, dm_srctrg: Any, logger: Any, checkpoint_path: str, trg_pseudo_label: torch.Tensor, trg_pseudo_label_index_map: Dict, kl_div_per_sample: Any) -> Tuple[Any, L.Trainer]:
    print(f"{bcolors.HEADER}=====> PERFORM DOMAIN ADAPTATION WITH TRG MODEL <===== setting {bcolors.ENDC}")
    print_ascii_art("DOMAIN ADAPTATION")
    print(f"{bcolors.HEADER}=====> PERFORM DOMAIN ADAPTATION WITH TRG MODEL <===== setting {bcolors.ENDC}")
    time.sleep(2)
    
    light_model_trg = import_trg_da_lightmodule(cfg, src_model=light_model.model, 
                                                pseudo_label=trg_pseudo_label,
                                                pseudo_label_index_map=trg_pseudo_label_index_map,
                                                kl_div_per_sample=kl_div_per_sample,)


    callbacks = setup_callbacks(cfg, checkpoint_path, is_source=False)
    trg_trainer = setup_trainer(cfg, callbacks, logger, is_source=False)
    # light_model_trg.freeze_modules()
    trg_trainer.fit(light_model_trg, dm_srctrg)
    
    return light_model_trg, trg_trainer

def evaluate_da_results(cfg: DictConfig, trg_trainer: L.Trainer, light_model_trg: Any, dm_srctrg: Any, metrics_calculator: MetricCalculator) -> None:
    """Evaluate the Domain Adaptation Results."""
    dm_srctrg.setup(stage="predict")
    print(f"Best model path: {trg_trainer.checkpoint_callback.best_model_path}")
    predict_trg_outputs_after_da = trg_trainer.predict(
        light_model_trg,
        dm_srctrg.test_dataloader(),
        ckpt_path=trg_trainer.checkpoint_callback.best_model_path,
        return_predictions=True,
    )
    metrics_calculator.save_metrics(predict_trg_outputs_after_da, "trg_test_results_da")


@hydra.main(version_base=None, config_path="da_conf", config_name="da_config")
def main(cfg: DictConfig) -> None:

    setup_environment(cfg)
    logger = setup_neptune_logger(cfg)
    # checkpoint_path = os.path.join("checkpoints/", str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")) + f"_{os.getpid()}")
    checkpoint_path = os.path.join(f"{cfg.save_output_path}/checkpoints")
    flush_checkpoint_path(cfg, checkpoint_path)

    callbacks = setup_callbacks(cfg, checkpoint_path, is_source=True)
    trainer = setup_trainer(cfg, callbacks, logger, is_source=True)

    # Import DataModules.
    dm_src = import_src_datamodule(cfg, log)
    dm_trg = import_trg_datamodule(cfg, log)
    dm_srctrg = import_srctrg_datamodule(cfg, log)

    if cfg.task.task_name == "train_from_src":
        print(f"{bcolors.HEADER}=====> TRAINING FROM SCRATCH <===== setting {bcolors.ENDC}")
        light_model, best_src_ckpt_path = train_source_model(cfg, trainer, dm_src)
    elif cfg.task.task_name == "train_from_pretrained":
        print(f"{bcolors.HEADER}=====> LOADING PRETRAINED MODEL <===== setting {bcolors.ENDC}")
        light_model, best_src_ckpt_path = load_source_model(cfg, dm_src)
    else:
        raise ValueError(f"Unknown task name: {cfg.task.task_name}")

    trans_matrix, metrics_calculator, predict_trg_train_outputs = calculate_transition_matrices(cfg, trainer, light_model, dm_src, dm_trg, best_src_ckpt_path)
    if cfg.likelihood.use_distance_weighting:
        distance_per_channel = calculate_distance_per_channels(cfg, trans_matrix, light_model)
    else:
        distance_per_channel = None
    trg_pseudo_label, trg_pseudo_label_index_map, kl_div_per_sample = get_pseudo_label_results(cfg, trans_matrix, predict_trg_train_outputs, distance_per_channel)
    # # setup Target Lightning Model For Pseudo-Labeling

    light_model_trg, trg_trainer = perform_domain_adaptation(cfg, light_model, dm_srctrg, logger, checkpoint_path, trg_pseudo_label, trg_pseudo_label_index_map, kl_div_per_sample)
    
    evaluate_da_results(cfg, trg_trainer, light_model_trg, dm_srctrg, metrics_calculator)

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    sys.argv.append("hydra.job.chdir=False")
    main()