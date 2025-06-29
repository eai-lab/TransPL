
import numpy as np
import pandas as pd
import glob
import torch


class TransitionMatrix:
    def __init__(self, cfg):
        self.cfg = cfg
        self.src_transition_matrix = None
        self.trg_transition_matrix = None
        self.src_classwise_transition_matrix = None
        self.coarse_codes_num = self.cfg.da_model.coarse_num_code
        self.save_output_path = cfg.save_output_path

        print("Initialized TransitionMatrix class.")
    
    def calculate_transition_matrix(self, predict_outputs):
        """
        Calculate the (Class-Agnostic) transition matrix from predict_outputs
        :param predict_outputs: Predicted outputs. [torch.tensor]
        :return: Transition matrix. [torch.tensor]
        """
        coarse_encodings_matrix = torch.cat([output['coarse_encoding_indices'] for output in predict_outputs])
        _, channel, _ = coarse_encodings_matrix.shape
        coarse_transition_matrix = torch.zeros((channel, self.coarse_codes_num, self.coarse_codes_num))

        # Process transitions for all samples and channels at once
        current_codes = coarse_encodings_matrix[:, :, :-1]
        next_codes = coarse_encodings_matrix[:, :, 1:]

        # Count transitions using a vectorized approach
        for c in range(channel):
            idx_current = current_codes[:, c].flatten()
            idx_next = next_codes[:, c].flatten()
            # Create a flat index for a 2D tensor of shape (self.coarse_codes_num, self.coarse_codes_num)
            flat_indices = idx_current * self.coarse_codes_num + idx_next
            # Count each unique index using bincount
            counts = torch.bincount(flat_indices, minlength=self.coarse_codes_num*self.coarse_codes_num)
            # Reshape to the size of the code transition matrix
            coarse_transition_matrix[c] = counts.view(self.coarse_codes_num, self.coarse_codes_num)
            
        # Add a small epsilon to prevent division by zero
        epsilon = 1e-8
        row_sums = coarse_transition_matrix.sum(dim=-1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=epsilon)
        # Normalize the transition matrix for each channel
        normalized_matrix = coarse_transition_matrix / row_sums

        # Replace any remaining NaN or Inf values with zeros
        normalized_matrix = torch.nan_to_num(normalized_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        return normalized_matrix
    
    def calculate_classwise_transition_matrix(self, predict_outputs):
        """
        Calculate the class-wise transition matrix from predict_outputs
        :param predict_outputs: Predicted outputs. [torch.tensor]
        :return: Transition matrix. [torch.tensor]
        """
        coarse_encodings_matrix = torch.cat([output['coarse_encoding_indices'] for output in predict_outputs])
        y_true = torch.cat([output['y_true'] for output in predict_outputs])
        y_true_unique_num = len(torch.unique(y_true))
        _, channel, _ = coarse_encodings_matrix.shape
        classwise_coarse_transition_matrix = torch.zeros((y_true_unique_num, channel, self.coarse_codes_num, self.coarse_codes_num))

        # Process transitions for all samples and channels at once
        current_codes = coarse_encodings_matrix[:, :, :-1] # (batch_size, channel, seq_len-1)
        next_codes = coarse_encodings_matrix[:, :, 1:] #(batch_size, channel, seq_len-1)

        # Count transitions using a vectorized approach for each class
        for class_idx in range(y_true_unique_num):
            class_mask = (y_true == class_idx)
            for c in range(channel):
                idx_current = current_codes[class_mask, c].flatten()
                idx_next = next_codes[class_mask, c].flatten()
                # Create a flat index for a 2D tensor of shape (self.coarse_codes_num, self.coarse_codes_num)
                flat_indices = idx_current * self.coarse_codes_num + idx_next
                # Count each unique index using bincount
                counts = torch.bincount(flat_indices, minlength=self.coarse_codes_num*self.coarse_codes_num)
                # Reshape to the size of the code transition matrix
                classwise_coarse_transition_matrix[class_idx, c] = counts.view(self.coarse_codes_num, self.coarse_codes_num)

        # Add a small epsilon to prevent division by zero
        epsilon = 1e-8
        row_sums = classwise_coarse_transition_matrix.sum(dim=-1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=epsilon)
        # Normalize the transition matrix for each class and channel
        normalized_matrix = classwise_coarse_transition_matrix / row_sums

        # check if there is a uniform 

        # Replace any remaining NaN or Inf values with zeros
        normalized_matrix = torch.nan_to_num(normalized_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        return normalized_matrix

        
    def calculate_src_transition_matrix(self, predict_outputs):
        """
        Calculate and store the source transition matrix
        :param predict_outputs: Predicted outputs. [torch.tensor]
        """
        print("Calculating source transition matrix.")
        self.src_transition_matrix = self.calculate_transition_matrix(predict_outputs)

    def calculate_src_classwise_transition_matrix(self, predict_outputs):
        """
        Calculate and store the source class-wise transition matrix
        :param predict_outputs: Predicted outputs. [torch.tensor]
        """
        print("Calculating source class-wise transition matrix.")
        self.src_classwise_transition_matrix = self.calculate_classwise_transition_matrix(predict_outputs)

    def calculate_trg_transition_matrix(self, predict_outputs):
        """
        Calculate and store the target transition matrix
        :param predict_outputs: Predicted outputs. [torch.tensor]
        """
        print("Calculating target transition matrix.")
        self.trg_transition_matrix = self.calculate_transition_matrix(predict_outputs)

    def get_src_transition_matrix(self):
        return self.src_transition_matrix
    
    def get_trg_transition_matrix(self):
        return self.trg_transition_matrix

    def load_src_transition_matrix(self):
        """
        Load the source transition matrix
        """
        assert self.cfg.task.task_name == "train_from_trg", "Only available for train_from_trg task."

        source_transition_matrix_path = self.cfg.task.source_model.source_transition_matrix_path
        src_transition_matrix = np.load(source_transition_matrix_path + "src_transition_matrix.npy")
        self.src_transition_matrix = torch.tensor(src_transition_matrix)
        print("Source transition matrix loaded.")

    def save_transition_matrices(self):
        """
        Save the source and target transition matrix
        """
        src_classwise_transition_matrix = self.src_classwise_transition_matrix.cpu().detach().numpy()
        src_transition_matrix = self.src_transition_matrix.cpu().detach().numpy()
        trg_transition_matrix = self.trg_transition_matrix.cpu().detach().numpy()
        
        np.save(self.save_output_path + "/src_classwise_transition_matrix.npy", src_classwise_transition_matrix)
        np.save(self.save_output_path + "/src_transition_matrix.npy", src_transition_matrix)
        np.save(self.save_output_path + "/trg_transition_matrix.npy", trg_transition_matrix)
        print("Transition matrix saved.")

    def save_src_test_transition_matrix(self, src_test_transition_matrix):
        """
        Save the source test transition matrix
        """
        src_test_transition_matrix = src_test_transition_matrix.cpu().detach().numpy()
        np.save(self.save_output_path + "/src_test_transition_matrix.npy", src_test_transition_matrix)
        print("Source test transition matrix saved.")


