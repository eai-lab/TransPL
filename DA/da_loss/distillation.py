import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.T = self.cfg.light_model_trg.kd.temperature

    def forward(self, student_logits, pseudo_label):
        # Knowledge distillation loss
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            pseudo_label,
            reduction='batchmean'
        ) * (self.T ** 2)

        return kd_loss
    

    