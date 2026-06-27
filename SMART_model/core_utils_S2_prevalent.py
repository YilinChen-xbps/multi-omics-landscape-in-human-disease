from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd

## train,val
def train(model, device, data_loader, optimizer,loss_criteria, batch_file_path, epoch, fold):
    model.train()
    train_loss = 0
    all_out_probs = []
    all_targets = []

    with open(batch_file_path, 'a') as f:
        f.write("Training Progress:\n")
        f.write(f"Fold {fold}, Epoch {epoch}\n")
        for batch, tensor in enumerate(tqdm(data_loader, desc="Training Progress")):
            tensor = [x.to(device) if hasattr(x, "to") else x for x in tensor]
            *inputs, target = tensor
            optimizer.zero_grad()
            out, *_ = model(*inputs)
            out_prob = torch.sigmoid(out)
            loss = loss_criteria(out.squeeze(1),target)
            if len(_) == 5:
                aux_loss = 0.3 * (
                    loss_criteria(_[0].squeeze(1), target) +
                    loss_criteria(_[1].squeeze(1), target) +
                    loss_criteria(_[2].squeeze(1), target) +
                    0.1 * loss_criteria(_[3].squeeze(1), target)
                )
                loss = loss + aux_loss
            if len(_) == 4:
                aux_loss = 0.3 * (
                    loss_criteria(_[0].squeeze(1), target) +
                    loss_criteria(_[1].squeeze(1), target) +
                    loss_criteria(_[2].squeeze(1), target)
                )
                loss = loss + aux_loss
            train_loss += loss.item()
            out_prob_np = out_prob.cpu().detach().numpy()
            target_np = target.cpu().detach().numpy()
            all_out_probs.append(out_prob_np)
            all_targets.append(target_np) 

            loss.backward()
            optimizer.step()
            f.write(f"Batch {batch + 1}: Train Loss = {loss.item():.6f}\n")
        avg_loss = train_loss / (batch+1)
        all_out_probs_np = np.concatenate(all_out_probs)
        all_targets_np = np.concatenate(all_targets)
        auc_score = roc_auc_score(all_targets_np, all_out_probs_np)
        print('Training set: Average loss: {:.6f}, AUC: {:.4f}\n'.format(avg_loss, auc_score))
        f.write(f"Fold {fold}, Epoch {epoch}: Average Train Loss: {avg_loss:.6f}, Train AUC: {auc_score:.4f}\n\n")
    return avg_loss,auc_score

def val(model, device,data_loader, loss_criteria,batch_file_path, epoch, fold,Test=False):
    model.eval()
    test_loss = 0
    all_out_probs = []
    all_targets = []

    with open(batch_file_path, 'a') as f:
        f.write("Validating Progress\n")
        f.write(f"Fold {fold}, Epoch {epoch}\n")
        with torch.no_grad():
            for batch, tensor in enumerate(tqdm(data_loader, desc="Validating Progress")):
                tensor = [x.to(device) if hasattr(x, "to") else x for x in tensor]
                *inputs, target = tensor
                out, *_ = model(*inputs)
                out_prob = torch.sigmoid(out)
                loss = loss_criteria(out.squeeze(1),target)
                test_loss += loss.item()
                out_prob_np = out_prob.cpu().detach().numpy()
                target_np = target.cpu().detach().numpy()
                all_out_probs.append(out_prob_np)
                all_targets.append(target_np) 
                
                f.write(f"Batch {batch + 1}: Val Loss = {loss.item():.6f}\n")
    
        avg_loss = test_loss/(batch+1)
        all_out_probs_np = np.concatenate(all_out_probs)
        all_targets_np = np.concatenate(all_targets)
        prob_saved = pd.DataFrame({
        'Logits': all_out_probs_np.flatten(),
        'target_y': all_targets_np
        })
        auc_score = roc_auc_score(all_targets_np, all_out_probs_np)
        print('Validation set: Average loss: {:.6f}, AUC: {:.4f}\n'.format(avg_loss, auc_score))
        f.write(f"Fold {fold}, Epoch {epoch}: Average Val Loss: {avg_loss:.6f}, Val AUC: {auc_score:.4f}\n\n")
    return avg_loss,auc_score,prob_saved