
import numpy as np
import pandas as pd
import glob
import torch
import torch.nn.functional as F
from tqdm import tqdm

torch.set_printoptions(sci_mode=False, precision=5)
from src.utils import bcolors
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)
from torchmetrics.functional.classification import multiclass_calibration_error

class LikelihoodEvaluator:
    def __init__(self, cfg, trans_matrix, distance_per_channel=None):
        self.cfg = cfg
        self.trans_matrix = trans_matrix
        self.pseudo_label = None
        self.likelihood_penalty = self.cfg.likelihood.likelihood_penalty
        print(f"{bcolors.OKGREEN}Likelihood penalty: {self.likelihood_penalty}{bcolors.ENDC}")
        self.pseudo_label_index_map = {}
        self.kl_div_per_sample = None
        self.use_trg_class_prior = self.cfg.likelihood.use_trg_class_prior
        if distance_per_channel is not None:
            print(f"{bcolors.OKGREEN}Using Gaussian Kernel to convert distance to similarity.{bcolors.ENDC}")
            self.sigma = self.cfg.likelihood.gaussian_sim.sigma
            self.similarity_per_channel = self.convert_distance_to_similarity(distance_per_channel)
        else:
            self.similarity_per_channel = None
            print(f"{bcolors.OKGREEN}Using raw distance for likelihood computation.{bcolors.ENDC}")
            

    # def calculate_likelihood(self, sequence, transition_matrix):
    # #Original likelihood calculation <Ours
    #     likelihood = 1.0
    #     for i in range(len(sequence) - 1):
    #         from_state = sequence[i]
    #         to_state = sequence[i + 1]
    #         likelihood *= transition_matrix[from_state][to_state] + 1e-6
    #     print(f"Likelihood: {likelihood}")
    #     return likelihood

    # def bayesian_posterior(self, class_num, channel_num, src_classwise_transition_matrix, trg_sequence_matrix, bayesian_prior):        
    #     bayesian_posterior = torch.zeros((channel_num, class_num))
    #     for channel_idx in range(channel_num):
    #         for class_idx in range(class_num):
    #             sequence_likelihood = self.calculate_likelihood(trg_sequence_matrix[channel_idx], src_classwise_transition_matrix[class_idx, channel_idx])
    #             bayesian_posterior[channel_idx, class_idx] = sequence_likelihood * bayesian_prior[class_idx]
    #     print(f"Bayesian Posterior: {bayesian_posterior}")
    #     evidence = torch.sum(bayesian_posterior, axis=1) + 1e-6 # Add a small epsilon to prevent division by zero
    #     bayesian_posterior = bayesian_posterior / evidence[:, None]

    #     # Normalize the posterior again, due to numerical instability
    #     bayesian_posterior = F.normalize(bayesian_posterior, p=1, dim=1)
    #     print(f"Normalized Bayesian Posterior: {bayesian_posterior}")
    #     return bayesian_posterior

    
    def calculate_log_likelihood(self, sequence, transition_matrix):
        """Calculate log likelihood using proper log-space arithmetic"""
        log_likelihood = 0.0
        seq_length = len(sequence) - 1
        
        for i in range(seq_length):
            from_state = sequence[i]
            to_state = sequence[i + 1]
            prob = transition_matrix[from_state][to_state]
            if prob <= 0:
                log_likelihood += torch.log(torch.tensor(self.likelihood_penalty))
            else:
                log_likelihood += torch.log(torch.tensor(prob))
        # print(f"log likelihood div by seq_length: {log_likelihood / seq_length}") # The more negative, the more unlikely.
        return log_likelihood / seq_length  # Return average log likelihood



    def bayesian_posterior(self, class_num, channel_num, src_classwise_transition_matrix, 
                        trg_sequence_matrix, bayesian_prior):
        # Convert prior to log space (tensor [0.2 0.2 0.2 0.2 0.2])
        # Divide by temperature
        # print(f"Log Prior : {torch.log(bayesian_prior)}")
        log_prior = torch.log(bayesian_prior) / self.cfg.likelihood.prior_temperature  # (tensor[-1.79  -1.79  -1.79  -1.79  -1.79])
        # print(f"Scaled Log Prior : {log_prior}")


        # Work in log space
        log_posterior = torch.zeros((channel_num, class_num))
        for channel_idx in range(channel_num):
            for class_idx in range(class_num):
                log_likelihood = self.calculate_log_likelihood(
                    trg_sequence_matrix[channel_idx],
                    src_classwise_transition_matrix[class_idx, channel_idx]
                )
                log_posterior[channel_idx, class_idx] = log_likelihood + log_prior[class_idx]
        
        # Convert back to probability space with proper normalization
        posterior = torch.exp(log_posterior)
        # print(f"Posterior: {posterior}")
        evidence = torch.sum(posterior, dim=1, keepdim=True)
        # Do we need to divide by evidence? it seems to strongly bias the posterior
        posterior = posterior / evidence
        # print(f"Posterior: {posterior}")
        
        return posterior


    def construct_pseudo_label(self, predict_outputs):
        src_classwise_transition_matrix = self.trans_matrix.src_classwise_transition_matrix  
        coarse_encodings_matrix = torch.cat([output['coarse_encoding_indices'] for output in predict_outputs])
        global_ids = torch.cat([output['gid'] for output in predict_outputs]).cpu().numpy().tolist()
        class_num, channel_num, code_num, _ = src_classwise_transition_matrix.shape

        assert len(global_ids) == coarse_encodings_matrix.shape[0], f"Global IDs: {len(global_ids)}, Encodings: {coarse_encodings_matrix.shape[0]} size mismatch."
        assert channel_num == coarse_encodings_matrix.shape[1], f"Channel number mismatch: {channel_num} != {coarse_encodings_matrix.shape[1]}"

        if self.use_trg_class_prior:
            class_prior_distribution = self.construct_prior_distribution(predict_outputs)
            print(f"{bcolors.OKGREEN}Using target class prior for pseudo label construction.{bcolors.ENDC}")
        else:
            # use uniform prior if we don't use trg class prior
            class_prior_distribution = torch.ones(src_classwise_transition_matrix.shape[0]) / class_num
            print(f"{bcolors.OKGREEN}Using uniform prior for pseudo label construction.{bcolors.ENDC}")
        print(f"Class prior distribution: {class_prior_distribution}")

        gid_pseudo_soft_label = []
        for trg_sample_idx in tqdm(range(coarse_encodings_matrix.shape[0]), desc="Constructing Pseudo Label"):
            trg_sequence_matrix = coarse_encodings_matrix[trg_sample_idx]
            bayesian_posterior = self.bayesian_posterior(class_num, channel_num, src_classwise_transition_matrix, trg_sequence_matrix, class_prior_distribution)
            if self.similarity_per_channel is not None:
                bayesian_posterior = self.weighted_average(bayesian_posterior)
            # print(f"weighted bayesian posterior: {bayesian_posterior}")
            bayesian_posterior_avg = torch.mean(bayesian_posterior, axis=0)
            bayesian_posterior_avg = F.normalize(bayesian_posterior_avg, p=1, dim=0)
            # print(f"Bayesian posterior avg: {bayesian_posterior_avg}")
            # print("================================================")
            # print(f"Bayesian posterior avg: {bayesian_posterior_avg}")
            if bayesian_posterior_avg.sum() < 0.9:
                print(f"{bcolors.WARNING}Warning: Pseudo label sum is less than 0.9. Set to uniform {bcolors.ENDC}")
                bayesian_posterior_avg = torch.ones_like(bayesian_posterior_avg) / class_num
            gid_pseudo_soft_label.append(bayesian_posterior_avg)

        
        self.pseudo_label = torch.stack(gid_pseudo_soft_label).to(coarse_encodings_matrix.device)
        self.pseudo_label_index_map = {gid: idx for idx, gid in enumerate(global_ids)}

        # Just to evaluate the pseudo labels
        self.save_pseudo_label_performance(predict_outputs)


        print("Pseudo labels constructed.")

    def weighted_average(self, bayesian_posterior):
        bayesian_posterior = self.similarity_per_channel * bayesian_posterior
        return bayesian_posterior

    def get_pseudo_label(self):
        return self.pseudo_label, self.pseudo_label_index_map, self.kl_div_per_sample
    
    def convert_distance_to_similarity(self, distance):
        # use gaussian kernel to convert distance to similarity
        # similarity = torch.exp(-distance / (2 * self.sigma ** 2)) # <- this is the original one, 28/10/2024
        similarity = torch.exp(-distance**2 / (2 * self.sigma ** 2)) # <- this is the new one, 28/10/2024
        print(f"Converted distance to similarity using Gaussian kernel with sigma: {self.sigma}")
        print(f"Similarity: {similarity}")
        return similarity.unsqueeze(1)

    
    def construct_prior_distribution(self, predict_outputs):
        y_true = torch.cat([output['y_true'] for output in predict_outputs]).cpu().numpy()
        class_prior_distribution = np.zeros(self.cfg.data.num_class)
        for class_idx in range(self.cfg.data.num_class):
            class_prior_distribution[class_idx] = np.sum(y_true == class_idx) + 1e-6
        class_prior_distribution = class_prior_distribution / np.sum(class_prior_distribution)
        return torch.tensor(class_prior_distribution)

    
    def save_pseudo_label_performance(self, predict_outputs):
        y_true = torch.cat([output['y_true'] for output in predict_outputs]).cpu().numpy()
        y_true_torch = torch.tensor(y_true)
        y_pred_proba = self.pseudo_label.cpu().numpy()
        y_pred = np.argmax(y_pred_proba, axis=1)
        # Measure KL divergence between uniform distribution and y_pred_proba for each sample
        kl_div = F.kl_div(
                            F.log_softmax(torch.tensor(y_pred_proba), dim=1),
                            torch.ones_like(torch.tensor(y_pred_proba)) / self.cfg.data.num_class,
                            reduction='none'
                        )

        # Sum across the class dimension (dim=1)
        kl_div_per_sample = kl_div.sum(dim=1, keepdim=True)
        # normalize y_pred_proba. There a numerical issue with the pseudo label
    
        # Calibration Error
        ece_score = multiclass_calibration_error(self.pseudo_label, y_true_torch, num_classes=self.cfg.data.num_class,
                                                    n_bins=10, norm='l1')
        mce_score = multiclass_calibration_error(self.pseudo_label, y_true_torch, num_classes=self.cfg.data.num_class,
                                                    n_bins=10, norm='max')
        
        accuracy = accuracy_score(y_true, y_pred)
        if self.cfg.data.num_class == 2:
            auroc = roc_auc_score(y_true, y_pred_proba[:, 1])
            auprc = average_precision_score(y_true, y_pred_proba[:, 1])
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
        else:
            # normalize y_pred_proba. There might be a numerical issue with the pseudo label
            y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
            auroc = roc_auc_score(y_true, y_pred_proba, multi_class="ovo")
            auprc = average_precision_score(y_true, y_pred_proba)
            f1 = f1_score(y_true, y_pred, average="macro")
            precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
            recall = recall_score(y_true, y_pred, average="macro")
        print("==== Pseudo Label Performance ====")
        print(f"Accuracy: {accuracy:.3f} AUROC: {auroc:.3f} AUPRC: {auprc:.3f} F1: {f1:.3f} Precision: {precision:.3f} Recall: {recall:.3f}")
        print(f"ECE: {ece_score.item():.3f} MCE: {mce_score.item():.3f}")
        pseudo_label_df = pd.DataFrame(
            {
                "accuracy": accuracy,
                "auroc": auroc,
                "auprc": auprc,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "ece": ece_score.item(),
                "mce": mce_score.item(),
            },
            index=[0],
        )
        pseudo_label_df['exp_num'] = self.cfg.exp_num 
        pseudo_label_df['seed'] = self.cfg.seed
        pseudo_label_df.to_csv(f"{self.cfg.save_output_path}/pseudo_label_performance.csv", index=False)

        # save self.similarity_per_channel
        if self.cfg.likelihood.use_distance_weighting:
            similarity_per_channel = pd.DataFrame(self.similarity_per_channel.cpu().numpy(), columns=[f"similarity_{i}" for i in range(self.similarity_per_channel.shape[1])])
            similarity_per_channel['exp_num'] = self.cfg.exp_num
            similarity_per_channel['seed'] = self.cfg.seed
            similarity_per_channel['channel'] = range(self.cfg.data.c_in)
            similarity_per_channel.to_csv(f"{self.cfg.save_output_path}/similarity_metrics.csv", index=False)

        # acutal pseudo label
        pseudo_label = pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "correct": y_true == y_pred,
            "kl_div": kl_div_per_sample.squeeze().cpu().numpy(),
            'accuracy': accuracy,
        })
        y_pred_proba_df = pd.DataFrame(y_pred_proba, columns=[f"prob_{i}" for i in range(self.cfg.data.num_class)])
        pseudo_label = pd.concat([pseudo_label, y_pred_proba_df], axis=1)
        pseudo_label['exp_num'] = self.cfg.exp_num
        pseudo_label['seed'] = self.cfg.seed
        pseudo_label.to_csv(f"{self.cfg.save_output_path}/pseudo_label.csv", index=False)

        self.kl_div_per_sample = kl_div_per_sample.squeeze()
