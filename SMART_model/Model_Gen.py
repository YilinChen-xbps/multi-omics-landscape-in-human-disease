import torch.nn as nn

class GEN_MLP2(nn.Module):
    def __init__(self, input_dim_g: int, hidden_size_g: int, dropout=0.25, n_classes=4,dim=32):
        super(GEN_MLP2, self).__init__() 
        self.fc_genomic = nn.Sequential(nn.Linear(input_dim_g, hidden_size_g), nn.ReLU(), nn.Dropout(p=dropout))
        self.fc = nn.Sequential(nn.Linear(hidden_size_g,dim), nn.ReLU(), nn.Dropout(p=dropout))
        self.classifier_g = nn.Linear(dim, n_classes) 
    def forward(self, g_x):
        g_features = self.fc(self.fc_genomic(g_x))
        g_logits = self.classifier_g(g_features)
        return g_logits, g_features