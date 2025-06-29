#!/bin/bash
START_EXP_NUM=100
data=wisdm

light_model_src=cls_light_myvqvae_double
light_model_trg=cls_light_myvqvae_double_adapt
da_model=cls_myvqvae_double
task=train_from_src
d_model=128
coarse_num_code=8
fine_num_code=64 # total_fine_num_code will be fine_num_code * channel.

batch_size=32
src_lr_rate=0.002
trg_lr_rate=0.0005
src_max_epochs=200
trg_max_epochs=200
trg_patience=5

kmean_init=true
gpu_id=1


logger_mode=debug
patch_len=8

src_trg_pairs=("7 18" "20 30" "35 31" "17 23" "6 19" "2 11" "33 12" "5 26" "4 28" "23 32")



feature_alignment=pot
gaussian_sim_sigma=0.1
use_distance_weighting=true

pseudo_label_confidence_topk_sampling=True
pseudo_topk_percent=0.5 # 0.1 0.2 0.5


for use_trg_class_prior in False True
do
    for pair in "${src_trg_pairs[@]}"
    do
        read -r src trg <<< "$pair"
        echo "Start domain adaptation experiment $START_EXP_NUM, $src -> $trg, patch len: $patch_len, kd lambda: $kd_lambda"
        CUDA_VISIBLE_DEVICES=$gpu_id python main.py data=$data data.da_data.src=$src  data.da_data.trg=$trg \
                                task=$task \
                                logger.mode=$logger_mode \
                                task.limit_train_batches=1.0 \
                                task.limit_val_batches=1.0 \
                                gpu_id=0 light_model_src=$light_model_src light_model_trg=$light_model_trg \
                                exp_num=$START_EXP_NUM \
                                da_model=$da_model \
                                da_model.patch_len=$patch_len \
                                da_model.coarse_num_code=$coarse_num_code da_model.fine_num_code=$fine_num_code \
                                da_model.d_model=$d_model \
                                da_model.coarse_kmeans_init=$kmean_init da_model.fine_kmeans_init=$kmean_init \
                                da_model.pseudo_label_confidence_topk_sampling=$pseudo_label_confidence_topk_sampling \
                                da_model.pseudo_topk_percent=$pseudo_topk_percent \
                                feature_alignment=$feature_alignment \
                                likelihood.use_trg_class_prior=$use_trg_class_prior \
                                likelihood.use_distance_weighting=$use_distance_weighting \
                                likelihood.gaussian_sim.sigma=$gaussian_sim_sigma \
                                light_model_src.optimizer.lr=$src_lr_rate \
                                light_model_src.callbacks.max_epochs=$src_max_epochs \
                                light_model_src.dataset.batch_size=$batch_size \
                                light_model_trg.optimizer.lr=$trg_lr_rate \
                                light_model_trg.callbacks.max_epochs=$trg_max_epochs \
                                light_model_trg.dataset.batch_size=$batch_size \
                                light_model_trg.callbacks.patience=$trg_patience \
                                seed=2020 deterministic=true
        echo "finish $START_EXP_NUM"
        START_EXP_NUM=$((START_EXP_NUM + 1))
    done
done

