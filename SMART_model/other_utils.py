import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import torch.nn as nn

class BCEWithLogitsLossIgnoreNaN(nn.BCEWithLogitsLoss):
    def forward(self, input, target):
        mask = ~torch.isnan(target)
        if not mask.any():
            raise ValueError("All target values are NaN. No valid targets to compute the loss.")

        masked_input = torch.masked_select(input, mask)
        masked_target = torch.masked_select(target, mask)

        loss = F.binary_cross_entropy_with_logits(
            masked_input,
            masked_target,
            weight=self.weight, 
            pos_weight=self.pos_weight, 
            reduction=self.reduction, 
        )

        return loss

class CoxPHLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_time, y_event):
        time = y_time
        event = y_event

        sort_time = torch.argsort(time, 0, descending=True)
        event = torch.gather(event, 0, sort_time)
        risk = torch.gather(y_pred.squeeze(), 0, sort_time)
        exp_risk = torch.exp(risk)
        log_risk = torch.log(torch.cumsum(exp_risk, 0))
        censored_likelihood = (risk - log_risk) * event
        censored_likelihood = torch.sum(censored_likelihood)
        censored_likelihood = censored_likelihood / y_time.shape[0]
        return -censored_likelihood

def calculate_masked_auc(y_true, y_pred):
    aucs = []
    for i in range(y_true.shape[1]):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]
        
        mask = ~np.isnan(col_true)
        valid_true = col_true[mask]
        valid_pred = col_pred[mask]
        
        if len(np.unique(valid_true)) == 2:
            score = roc_auc_score(valid_true, valid_pred)
            aucs.append(score)
        else:
            continue 
            
    return np.mean(aucs) if aucs else None

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, path='best_model.pt', delta=0,save_best_record=False, save_best_logits=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path
        self.delta = delta
        self.save_best_record = save_best_record
        self.save_best_logits = save_best_logits
        if self.save_best_record:
            self.best_record = None
        if self.save_best_logits:
            self.best_prob_saved = None

    def __call__(self, valid_loss, model, record=None, logits=None):
        if self.best_loss is None:
            self.save_checkpoint(valid_loss, model, record, logits)
            self.best_loss = valid_loss
        
        elif valid_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        
        else:
            self.save_checkpoint(valid_loss, model, record, logits)
            self.best_loss = valid_loss
            self.counter = 0

    def save_checkpoint(self, valid_loss, model, record, logits):
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss if self.best_loss else float("inf"):.6f} --> {valid_loss:.6f}). Saving model...')
        
        torch.save(model.state_dict(), self.path)
        if self.save_best_record:
            self.best_record = record
        if self.save_best_logits:
            self.best_prob_saved = logits.detach().cpu()