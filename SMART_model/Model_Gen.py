import math
import torch
import torch.nn as nn

class MM_MLP(nn.Module):
    def __init__(self, input_dim_g: int, hidden_size_g: int, dropout=0.25, n_classes=4):
        super(MM_MLP, self).__init__()  # Inherited from the parent class nn.Module
        self.fc_genomic = nn.Sequential(nn.Linear(input_dim_g, hidden_size_g), nn.ReLU(), nn.Dropout(p=dropout))
        #self.fc_mm = nn.Sequential(nn.Linear(hidden_size_g+hidden_size_h, hidden_size_mm), nn.ReLU(), nn.Dropout(p=dropout))
        self.classifier_g = nn.Linear(hidden_size_g, n_classes)  # 2nd Full-Connected Layer: hidden node -> output
        #self.classifier = nn.Linear(hidden_size_mm, n_classes)  # 2nd Full-Connected Layer: hidden node -> output

    def forward(self, g_x):
        g_features = self.fc_genomic(g_x)
        #mm_features = self.fc_mm(torch.cat([g_features, h_features], dim=-1))
        g_logits = self.classifier_g(g_features)
        g_probs = torch.softmax(g_logits, dim=-1)
        #logits = self.classifier(mm_features)
        #probs = torch.softmax(logits, dim=-1)
        #return g_logits, g_features, g_probs
        return g_logits, g_features

class MM_MLP2(nn.Module):
    def __init__(self, input_dim_g: int, hidden_size_g: int, dropout=0.25, n_classes=4,dim=32):
        super(MM_MLP2, self).__init__()  # Inherited from the parent class nn.Module
        self.fc_genomic = nn.Sequential(nn.Linear(input_dim_g, hidden_size_g), nn.ReLU(), nn.Dropout(p=dropout))
        #self.fc_mm = nn.Sequential(nn.Linear(hidden_size_g+hidden_size_h, hidden_size_mm), nn.ReLU(), nn.Dropout(p=dropout))
        self.fc = nn.Sequential(nn.Linear(hidden_size_g,dim), nn.ReLU(), nn.Dropout(p=dropout))
        self.classifier_g = nn.Linear(dim, n_classes)  # 2nd Full-Connected Layer: hidden node -> output
        #self.classifier = nn.Linear(hidden_size_mm, n_classes)  # 2nd Full-Connected Layer: hidden node -> output

    def forward(self, g_x):
        g_features = self.fc(self.fc_genomic(g_x))
        #mm_features = self.fc_mm(torch.cat([g_features, h_features], dim=-1))
        g_logits = self.classifier_g(g_features)
        g_probs = torch.softmax(g_logits, dim=-1)
        #logits = self.classifier(mm_features)
        #probs = torch.softmax(logits, dim=-1)
        #return g_logits, g_features, g_probs
        return g_logits, g_features