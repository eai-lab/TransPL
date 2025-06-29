import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange



class MyVectorQuantizer(nn.Module):
    def __init__(self, cfg, patch_len, code_dim, num_code, commitment_cost):
        super(MyVectorQuantizer, self).__init__()
        self.cfg = cfg 
        self.patch_len = patch_len
        self.code_dim = code_dim
        self.num_code = num_code
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(self.num_code, self.code_dim)
        self.codebook.weight.data.uniform_(-1 / self.num_code, 1 / self.num_code)

        
    def forward(self, encoded_patch_input):
        # Batch, Channel, Time (Patch Num), d_model
        B, C, T, D = encoded_patch_input.shape
        assert D == self.code_dim, f"Expected input dimension to be {self.code_dim}, not {D}"

        encoded_patch_input = F.normalize(encoded_patch_input, p=2, dim=-1)
        codebook = F.normalize(self.codebook.weight, p=2, dim=-1) # [num_code, code_dim]

        # reshape encoded_patch_input
        # encoded_patch_input = rearrange(encoded_patch_input, 'b c t d -> (b c)

        similarity = torch.einsum('bctd,nd->bctn', encoded_patch_input, codebook)
        code_indices = torch.argmax(similarity, dim=-1)

        # Create one-hot encodings from the code indices
        one_hot_encodings = F.one_hot(code_indices, num_classes=self.num_code).float() # [B, C, T, num_code]

        # Quantize the inputs by using the one-hot encoded indices to gather from the codebook
        quantized = torch.einsum('bctn,nd->bctd', one_hot_encodings, codebook)

        # Compute losses
        e_latent_loss = F.mse_loss(quantized.detach(), encoded_patch_input)  # Encoders gradient is not propagated
        q_latent_loss = F.mse_loss(quantized, encoded_patch_input.detach())  # Gradient is only propagated through quantized
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Use the quantized values to calculate the output
        quantized = encoded_patch_input + (quantized - encoded_patch_input).detach()  # Straight-through estimator

        # Calculate the average probability of each code to determine perplexity
        avg_probs = torch.mean(one_hot_encodings, dim=[0, 1, 2])  # Average over all but the codebook dimension
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        # low perplexity means indicates that a small number of codes are frequently used,
        # while a high perplexity indicates that many codes are used with similar frequency.

        return loss, quantized, perplexity, self.codebook.weight, code_indices
