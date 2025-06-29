import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss

from einops import rearrange
from src.utils import bcolors

class CoarseQuantizer(nn.Module):
    def __init__(self, cfg, patch_len, code_dim, num_code, commitment_cost, kmeans_init=False):
        super(CoarseQuantizer, self).__init__()
        self.cfg = cfg
        self.patch_len = patch_len
        self.code_dim = code_dim
        self.num_code = num_code
        self.commitment_cost = commitment_cost
        

        self.codebook = nn.Embedding(self.num_code, self.code_dim)
        self.codebook.weight.data.uniform_(-1 / self.num_code, 1 / self.num_code)

        self.kmeans_init = kmeans_init
        self._kmeans_initialized = False
        if self.kmeans_init:
            print(f"KMeans initialization for CoarseQuantizer")
            self.kmeans = FaissKMeans(self.num_code, self.code_dim)

    def forward(self, encoded_patch_input):
        # Batch, Channel, Time (Patch Num), d_model
        B, C, T, D = encoded_patch_input.shape
        assert (
            D == self.code_dim
        ), f"Expected input dimension to be {self.code_dim}, not {D}"

        encoded_patch_input = F.normalize(encoded_patch_input, p=2, dim=-1)

        # Initialize kmeans only once during training
        if self.training and self.kmeans_init and not self._kmeans_initialized:
            centroids = self.kmeans.get_centroids(encoded_patch_input)
            self.codebook.weight.data = centroids
            self._kmeans_initialized = True

            
        codebook = F.normalize(
            self.codebook.weight, p=2, dim=-1
        )  # [num_code, code_dim]

        similarity = torch.einsum("bctd,nd->bctn", encoded_patch_input, codebook)
        code_indices = torch.argmax(similarity, dim=-1)

        # Create one-hot encodings from the code indices
        one_hot_encodings = F.one_hot(
            code_indices, num_classes=self.num_code
        ).float()  # [B, C, T, num_code]

        # Quantize the inputs by using the one-hot encoded indices to gather from the codebook
        quantized = torch.einsum("bctn,nd->bctd", one_hot_encodings, codebook)

        # Compute losses
        e_latent_loss = F.mse_loss(
            quantized.detach(), encoded_patch_input
        )  # Encoders gradient is not propagated
        q_latent_loss = F.mse_loss(
            quantized, encoded_patch_input.detach()
        )  # Gradient is only propagated through quantized
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Use the quantized values to calculate the output
        quantized = (
            encoded_patch_input + (quantized - encoded_patch_input).detach()
        )  # Straight-through estimator

        # Calculate the average probability of each code to determine perplexity
        avg_probs = torch.mean(
            one_hot_encodings, dim=[0, 1, 2]
        )  # Average over all but the codebook dimension
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, self.codebook.weight, code_indices


class FineQuantizer(nn.Module):
    def __init__(self, cfg, patch_len, code_dim, num_code, commitment_cost, kmeans_init=False):
        super(FineQuantizer, self).__init__()
        self.cfg = cfg
        self.patch_len = patch_len
        self.code_dim = code_dim
        self.num_code = num_code
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(self.num_code, self.code_dim)
        self.codebook.weight.data.uniform_(-1 / self.num_code, 1 / self.num_code)

        self.kmeans_init = kmeans_init
        self._kmeans_initialized = False
        if self.kmeans_init:
            print(f"KMeans initialization for FineQuantizer")
            self.kmeans = FaissKMeans(self.num_code, self.code_dim)

    def forward(self, encoded_patch_input):
        # Batch, Channel, Time (Patch Num), d_model
        B, C, T, D = encoded_patch_input.shape
        assert (
            D == self.code_dim
        ), f"Expected input dimension to be {self.code_dim}, not {D}"

        encoded_patch_input = F.normalize(encoded_patch_input, p=2, dim=-1)

        # Initialize kmeans only once during training
        if self.training and self.kmeans_init and not self._kmeans_initialized:
            print(f"KMeans initialization for FineQuantizer!!!!!")
            centroids = self.kmeans.get_centroids(encoded_patch_input)
            self.codebook.weight.data = centroids
            self._kmeans_initialized = True
            
        codebook = F.normalize(
            self.codebook.weight, p=2, dim=-1
        )  # [num_code, code_dim]

        similarity = torch.einsum("bctd,nd->bctn", encoded_patch_input, codebook)
        code_indices = torch.argmax(similarity, dim=-1)

        # Create one-hot encodings from the code indices
        one_hot_encodings = F.one_hot(
            code_indices, num_classes=self.num_code
        ).float()  # [B, C, T, num_code]

        # Quantize the inputs by using the one-hot encoded indices to gather from the codebook
        quantized = torch.einsum("bctn,nd->bctd", one_hot_encodings, codebook)

        # Compute losses
        e_latent_loss = F.mse_loss(
            quantized.detach(), encoded_patch_input
        )  # Encoders gradient is not propagated
        q_latent_loss = F.mse_loss(
            quantized, encoded_patch_input.detach()
        )  # Gradient is only propagated through quantized
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Use the quantized values to calculate the output
        quantized = (
            encoded_patch_input + (quantized - encoded_patch_input).detach()
        )  # Straight-through estimator

        # Calculate the average probability of each code to determine perplexity
        avg_probs = torch.mean(
            one_hot_encodings, dim=[0, 1, 2]
        )  # Average over all but the codebook dimension
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, self.codebook.weight, code_indices


# class FineQuantizer(nn.Module):
#     def __init__(self, cfg, patch_len, code_dim, num_code, commitment_cost, kmeans_init=False, trainable_channels=None):
#         super(FineQuantizer, self).__init__()
#         self.cfg = cfg
#         self.patch_len = patch_len
#         self.code_dim = code_dim
#         self.num_code = num_code
#         self.num_channels = self.cfg.data.c_in
#         self.commitment_cost = commitment_cost
#         self.trainable_channels = trainable_channels

#         # Create a separate Parameter for each channel
#         self.codebooks = nn.ParameterList([
#             nn.Parameter(torch.randn(self.num_code, self.code_dim))
#             for _ in range(self.num_channels)
#         ])

#         # Initialize the codebooks
#         for codebook in self.codebooks:
#             nn.init.uniform_(codebook, -1 / self.num_code, 1 / self.num_code)

#         self.kmeans_init = kmeans_init
#         self._kmeans_initialized = False
#         if self.kmeans_init:
#             print(f"KMeans initialization for FineQuantizer")
#             self.kmeans = [FaissKMeans(self.num_code, self.code_dim) for _ in range(self.num_channels)]

#     def freeze_all_channels(self):
#         self.set_trainable_channels([])
    
#     def unfreeze_all_channels(self):
#         self.set_trainable_channels(list(range(self.num_channels)))
    
#     def unfreeze_non_aligned_layers(self):
#         if self.trainable_channels is None:
#             raise ValueError("trainable_channels must be set before calling this method.")
#         else:
#             self.set_trainable_channels(self.trainable_channels)

#     def set_trainable_channels(self, trainable_channels=None):
#         if trainable_channels is None:
#             trainable_channels = list(range(self.num_channels))
        
#         for i, codebook in enumerate(self.codebooks):
#             codebook.requires_grad = (i in trainable_channels)

#         # print requires_grad status of each codebook
#         for i, codebook in enumerate(self.codebooks):
#             if codebook.requires_grad:
#                 print(f"{bcolors.OKGREEN}codebook for Channel {i} is trainable{bcolors.ENDC}")
#             else:
#                 print(f"{bcolors.OKCYAN}codebook for Channel {i} is frozen{bcolors.ENDC}")

#     def forward(self, encoded_patch_input):
#         B, C, T, D = encoded_patch_input.shape
#         assert D == self.code_dim, f"Expected input dimension to be {self.code_dim}, not {D}"
#         assert C == self.num_channels, f"Expected {self.num_channels} channels, got {C}"

#         encoded_patch_input = F.normalize(encoded_patch_input, p=2, dim=-1)

#         if self.training and self.kmeans_init and not self._kmeans_initialized:
#             if self.trainable_channels is None:
#                 print(f"KMeans initialization for FineQuantizer!, Initializing all channels")
#                 for c in range(self.num_channels):
#                     centroids = self.kmeans[c].get_centroids(encoded_patch_input[:, c], channel_specific=True)
#                     self.codebooks[c].data.copy_(centroids)
#             else:
#                 print(f"KMeans initialization for FineQuantizer!, Initializing trainable channels")
#                 for c in self.trainable_channels:
#                     centroids = self.kmeans[c].get_centroids(encoded_patch_input[:, c], channel_specific=True)
#                     self.codebooks[c].data.copy_(centroids)
#             self._kmeans_initialized = True

#         # Normalize each channel's codebook
#         normalized_codebooks = [F.normalize(codebook, p=2, dim=-1) for codebook in self.codebooks]
        
#         # Compute similarity for each channel separately
#         similarities = [torch.einsum('btd,nd->btn', encoded_patch_input[:, c], codebook) 
#                         for c, codebook in enumerate(normalized_codebooks)]
        
#         code_indices = torch.stack([torch.argmax(sim, dim=-1) for sim in similarities], dim=1)

#         # Create one-hot encodings from the code indices
#         one_hot_encodings = F.one_hot(code_indices, num_classes=self.num_code).float()  # [B, C, T, num_code]

#         # Quantize the inputs by using the one-hot encoded indices to gather from the codebook
#         quantized = torch.stack([
#             torch.einsum('btn,nd->btd', one_hot_encodings[:, c], codebook)
#             for c, codebook in enumerate(normalized_codebooks)
#         ], dim=1)


#         # Compute losses
#         e_latent_loss = F.mse_loss(quantized.detach(), encoded_patch_input)
#         q_latent_loss = F.mse_loss(quantized, encoded_patch_input.detach())
#         loss = q_latent_loss + self.commitment_cost * e_latent_loss

#         # Use the quantized values to calculate the output
#         quantized = encoded_patch_input + (quantized - encoded_patch_input).detach()  # Straight-through estimator

#         # Calculate the average probability of each code to determine perplexity
#         avg_probs = torch.mean(one_hot_encodings, dim=[0, 1, 2])  # Average over all but the codebook dimension
#         perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

#         # Convert ParameterList to a tensor for returning
#         codebook_tensor = torch.stack([codebook.data for codebook in self.codebooks])

#         return loss, quantized, perplexity, codebook_tensor, code_indices



class FaissKMeans:
    def __init__(self, num_code, code_dim, niter=50):
        self.num_code = num_code
        self.code_dim = code_dim
        self.niter = niter
        self.kmeans = faiss.Kmeans(
            d=self.code_dim,
            k=self.num_code,
            niter=self.niter,
            verbose=True,
            gpu=True,
        )

    def _fit(self, x):
        assert len(x.shape) == 4, "input shape must be (batch, channel, patch_num, embedding_dim)"
        with torch.no_grad():
            # Create a copy of x and move it to CPU
            x_copy = x.detach().cpu()
            # Rearrange and convert to numpy for faiss
            x_numpy = rearrange(x_copy, 'b c p e -> (b c p) e').numpy()
            # Train KMeans
            self.kmeans.train(x_numpy)
            # Convert centroids back to tensor and move to x's device
            self.centroids = torch.tensor(self.kmeans.centroids, device=x.device)
        print(f"centroids calculated from faiss: {self.centroids}")
    
    def _fit_channel_specific(self, x):
        assert len(x.shape) == 3, "input shape must be (batch, patch_num, embedding_dim)"
        with torch.no_grad():
            # Create a copy of x and move it to CPU
            x_copy = x.detach().cpu()
            # Rearrange and convert to numpy for faiss
            x_numpy = rearrange(x_copy, 'b p e -> (b p) e').numpy()
            # Train KMeans
            self.kmeans.train(x_numpy)
            # Convert centroids back to tensor and move to x's device
            self.centroids = torch.tensor(self.kmeans.centroids, device=x.device)
        print(f"centroids calculated from faiss: {self.centroids}")

    def get_centroids(self, X, channel_specific=False):
        if channel_specific:
            self._fit_channel_specific(X)
        else:
            self._fit(X)
        return self.centroids