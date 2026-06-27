import torch
import torch.nn as nn
from typing import Optional

class MM_MLP2(nn.Module):
    def __init__(self, input_dim_g: int, hidden_size_g: int, dropout: int, n_classes: int,dim: int,input_dim_cov: Optional[int] = None):
        super(MM_MLP2, self).__init__()  # Inherited from the parent class nn.Module
        self.fc_genomic = nn.Sequential(nn.Linear(input_dim_g, hidden_size_g), nn.ReLU(), nn.Dropout(p=dropout))
        self.fc = nn.Sequential(nn.Linear(hidden_size_g,dim), nn.ReLU(), nn.Dropout(p=dropout))
        self.classifier_g = nn.Linear(dim, n_classes)  # 2nd Full-Connected Layer: hidden node -> output
        if input_dim_cov is not None:
            self.fc_cov = nn.Sequential(nn.Linear(input_dim_cov, dim), nn.ReLU())
            fc_mm_in = dim * 2   # omic + cov
            self.fc_mm = nn.Sequential(nn.Linear(fc_mm_in, dim), nn.ReLU(), nn.Dropout(p=dropout))
        else:
            self.fc_cov = None # omic

    def forward(self, g_x, cov_x=None):
        g_features = self.fc(self.fc_genomic(g_x))
        if cov_x is not None:
            cov_features = self.fc_cov(cov_x)
            g_features = torch.cat([g_features, cov_features], dim=1)
            g_features = self.fc_mm(g_features)
        g_logits = self.classifier_g(g_features)
        return g_logits, g_features