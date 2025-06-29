import torch 
import torch.nn as nn
import torch.nn.functional as F




## BERT_Enc.py
class Trans_Conv(nn.Module):
    def __init__(self, d_model: int, dropout: float, in_channels: int, out_channels: int, activation: str='gelu',
                 **kwargs):
        super(Trans_Conv, self).__init__(**kwargs)

        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=8,
            dim_feedforward=4*d_model,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

    
    def forward(self, x: torch.Tensor):
        # x: (B, C, T, d_model)
        B, C, T, D = x.shape

        # get the feature
        x = x.contiguous().view(B*C, T, D)
        x = self.transformer(x)
        x = x.view(B, C, T, D) # (B, C, T, d_model)

        return x
    


### Quantize.py
class Quantize(nn.Module):
    def __init__(self, cfg, input_dim: int, vq_dim: int, num_embed: int, codebook_num: int=1, split_num:int=4, **kwargs):
        super(Quantize, self).__init__(**kwargs)
        # hyper-parameters
        self.cfg = cfg
        self.split_num = split_num

        self.linear = torch.arange(input_dim//2 + 1).to(f"cuda:{self.cfg.gpu_id}")
        # if input_dim is odd, we need to add one more dimension
        self.projector = torch.nn.init.xavier_normal_(torch.empty(input_dim, vq_dim))
        self.projector = nn.Parameter(self.projector, requires_grad=False)

        codebook = torch.nn.init.normal_(torch.empty(codebook_num, num_embed, vq_dim))
        self.codebook = nn.Parameter(codebook, requires_grad=False)

        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [batch * channel, num_patch, patch_dim]
        with torch.no_grad():
            x_fft = torch.fft.rfft(x, dim=-1) #x_fft [batch * channel, num_patch, patch_dim // 2 + 1] 
            magnitude = torch.abs(x_fft)
            phase = torch.angle(x_fft)

            # Ensure linear tensor matches the last dimension of x_fft
            phase_adjustment = phase[:, :, 1:2] * self.linear[:phase.shape[-1]]
            phase -= phase_adjustment

            x_recon = magnitude * torch.exp(1j * phase)
            x_recon = torch.fft.irfft(x_recon, n=x.shape[-1], dim=-1).unsqueeze(-2)  # Correctly specify 'n' to match original last dimension

            x_feature = x_recon.matmul(self.projector) # [bs, T, 1, vq_dim]

            x_feature_norm = x_feature.norm(dim=-1, keepdim=True)
            x_feature = x_feature / x_feature_norm # [bs, T, 1, vq_dim]
            codebook_norm = self.codebook.norm(dim=-1, keepdim=True)
            codebook = self.codebook / codebook_norm # [codebook_num, num_embed, vq_dim]

            similarity = torch.einsum('btnd,cmd->btncm', x_feature, codebook) # [bs, T, 1, codebook_num, num_embed]
            idx = torch.argmax(similarity, dim=-1) # [bs, T, 1, codebook_num]
            return idx
