import torch
import torch.nn as nn
from FTTransformer_S1 import FTTransformer
from typing import Literal,Optional
from Model_Gen import MM_MLP2

_LINFORMER_KV_COMPRESSION_SHARING = Literal['headwise', 'key-value']

class MultiOmicsFusion(nn.Module):
    def __init__(
        self,
        input_dim_g,
        input_dim_p,
        input_dim_m,
        input_dim_c,
        dim,
        dim_out,
        depth,
        heads,
        attn_dropout = 0.,
        ff_dropout = 0.,
        dropout = 0.,
        linformer_kv_compression_ratio: Optional[float] = None,
        linformer_kv_compression_sharing: Optional[
            _LINFORMER_KV_COMPRESSION_SHARING
        ] = None,
        input_dim_cov: Optional[int] = None,  
        ):
        super(MultiOmicsFusion, self).__init__()

        self.fc_gen = MM_MLP2(input_dim_g=input_dim_g,hidden_size_g=1024,dropout=dropout,n_classes=dim_out)
        self.fc_pro = FTTransformer(num_continuous = input_dim_p, dim = dim, dim_out = dim_out, depth = depth, heads = heads, attn_dropout = attn_dropout, ff_dropout = ff_dropout,
                                   n_tokens = input_dim_p+1, linformer_kv_compression_ratio = linformer_kv_compression_ratio, linformer_kv_compression_sharing = linformer_kv_compression_sharing)
        self.fc_met = FTTransformer(num_continuous = input_dim_m, dim = dim, dim_out = dim_out, depth = depth, heads = heads, attn_dropout = attn_dropout, ff_dropout = ff_dropout,
                                   n_tokens = input_dim_m+1, linformer_kv_compression_ratio = linformer_kv_compression_ratio, linformer_kv_compression_sharing = linformer_kv_compression_sharing)
        self.fc_che = FTTransformer(num_continuous = input_dim_c, dim = dim, dim_out = dim_out, depth = depth, heads = heads, attn_dropout = attn_dropout, ff_dropout = ff_dropout,
                                   n_tokens = input_dim_c+1, linformer_kv_compression_ratio = linformer_kv_compression_ratio, linformer_kv_compression_sharing = linformer_kv_compression_sharing)

        if input_dim_cov is not None:
            self.fc_cov = nn.Sequential(nn.Linear(input_dim_cov, dim), nn.ReLU())
            fc_mm_in = dim * 8   # gen + pro + met + che + prior(3*dim) + cov
        else:
            self.fc_cov = None
            fc_mm_in = dim * 7   # gen + pro + met + che + prior(3*dim)

        self.fc_mm = nn.Sequential(nn.Linear(fc_mm_in, dim), nn.ReLU(), nn.Dropout(p=dropout))
        self.classifier = nn.Linear(dim, dim_out)

    def forward(self, p_x, c_x, m_x, g_x, prior_x, cov_x=None):
        gen_logits, gen_features = self.fc_gen(g_x)
        pro_logits, pro_features, _ = self.fc_pro(p_x)
        met_logits, met_features, _ = self.fc_met(m_x)
        che_logits, che_features, _ = self.fc_che(c_x)

        parts = [gen_features, pro_features, met_features, che_features, prior_x]
        if self.fc_cov is not None and cov_x is not None:
            parts.append(self.fc_cov(cov_x))

        mm_features = self.fc_mm(torch.cat(parts, dim=-1))
        logits = self.classifier(mm_features)
        return logits, pro_logits, met_logits, che_logits, gen_logits, mm_features
        
class MultiOmicsFusionNoGen(nn.Module):
    def __init__(
        self,
        input_dim_p,
        input_dim_m,
        input_dim_c,
        dim,
        dim_out,
        depth,
        heads,
        attn_dropout = 0.,
        ff_dropout = 0.,
        dropout = 0.,
        linformer_kv_compression_ratio: Optional[float] = None,
        linformer_kv_compression_sharing: Optional[
            _LINFORMER_KV_COMPRESSION_SHARING
        ] = None,
        input_dim_cov: Optional[int] = None,  
        ):
        super(MultiOmicsFusionNoGen, self).__init__()

        self.fc_pro = FTTransformer(num_continuous = input_dim_p, dim = dim, dim_out = dim_out, depth = depth, heads = heads, attn_dropout = attn_dropout, ff_dropout = ff_dropout,
                                   n_tokens = input_dim_p+1, linformer_kv_compression_ratio = linformer_kv_compression_ratio, linformer_kv_compression_sharing = linformer_kv_compression_sharing)
        self.fc_met = FTTransformer(num_continuous = input_dim_m, dim = dim, dim_out = dim_out, depth = depth, heads = heads, attn_dropout = attn_dropout, ff_dropout = ff_dropout,
                                   n_tokens = input_dim_m+1, linformer_kv_compression_ratio = linformer_kv_compression_ratio, linformer_kv_compression_sharing = linformer_kv_compression_sharing)
        self.fc_che = FTTransformer(num_continuous = input_dim_c, dim = dim, dim_out = dim_out, depth = depth, heads = heads, attn_dropout = attn_dropout, ff_dropout = ff_dropout,
                                   n_tokens = input_dim_c+1, linformer_kv_compression_ratio = linformer_kv_compression_ratio, linformer_kv_compression_sharing = linformer_kv_compression_sharing)
        
        if input_dim_cov is not None:
            self.fc_cov = nn.Sequential(nn.Linear(input_dim_cov, dim), nn.ReLU())
            fc_mm_in = dim * 7   # pro + met + che + prior(3*dim) + cov
        else:
            self.fc_cov = None
            fc_mm_in = dim * 6   # pro + met + che + prior(3*dim)

        self.fc_mm = nn.Sequential(nn.Linear(fc_mm_in, dim), nn.ReLU(), nn.Dropout(p=dropout))
        self.classifier = nn.Linear(dim, dim_out)

    def forward(self, p_x, c_x, m_x, prior_x, cov_x=None):
        pro_logits, pro_features, _ = self.fc_pro(p_x)
        met_logits, met_features, _ = self.fc_met(m_x)
        che_logits, che_features, _ = self.fc_che(c_x)
        parts = [pro_features, met_features, che_features, prior_x]
        if self.fc_cov is not None and cov_x is not None:
            parts.append(self.fc_cov(cov_x))
        mm_features = self.fc_mm(torch.cat(parts, dim=-1))
        logits = self.classifier(mm_features)
        return logits, pro_logits, met_logits, che_logits, mm_features
        
class SingleOmics(nn.Module):
    def __init__(
        self,
        input_dim,
        dim,
        dim_out,
        depth,
        heads,
        attn_dropout = 0.,
        ff_dropout = 0.,
        dropout = 0.,
        n_tokens: Optional[int] = None,
        linformer_kv_compression_ratio: Optional[float] = None,
        linformer_kv_compression_sharing: Optional[
            _LINFORMER_KV_COMPRESSION_SHARING
        ] = None,
        input_dim_cov: Optional[int] = None,
        ):
        super(SingleOmics, self).__init__()  # Inherited from the parent class nn.Module

        self.fc_omic = FTTransformer(num_continuous = input_dim, dim = dim, dim_out = dim_out, depth = depth, heads = heads, attn_dropout = attn_dropout, ff_dropout = ff_dropout,
                                   n_tokens = input_dim+1, linformer_kv_compression_ratio = linformer_kv_compression_ratio, linformer_kv_compression_sharing = linformer_kv_compression_sharing)
        
        if input_dim_cov is not None:
            self.fc_cov = nn.Sequential(nn.Linear(input_dim_cov, dim), nn.ReLU())
            fc_mm_in = dim * 3   # omic + cov
        else:
            self.fc_cov = None
            fc_mm_in = dim * 2   # omic

        self.fc_mm = nn.Sequential(nn.Linear(fc_mm_in, dim), nn.ReLU(), nn.Dropout(p=dropout))
        self.classifier = nn.Linear(dim, dim_out)  # 2nd Full-Connected Layer: hidden node -> output

    def forward(self, omic_x, prior_x, cov_x=None):
        _,omic_features,_ = self.fc_omic(omic_x)
        prior_features = prior_x
        parts = [omic_features, prior_features]
        if self.fc_cov is not None and cov_x is not None:
            parts.append(self.fc_cov(cov_x))
        mm_features = self.fc_mm(torch.cat(parts, dim=-1))
        logits = self.classifier(mm_features)
        return logits, mm_features