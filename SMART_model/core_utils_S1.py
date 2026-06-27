from other_utils import calculate_masked_auc
from tqdm import tqdm
import torch
import numpy as np

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
            data, target = tensor
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            out, _ = model(data,None) #out 1024x1
            out_prob = torch.sigmoid(out)
            loss = loss_criteria(out,target)
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
        auc_score = calculate_masked_auc(all_targets_np, all_out_probs_np)
        print('Training set: Average loss: {:.6f}, AUC: {:.4f}\n'.format(avg_loss, auc_score))
        f.write(f"Fold {fold}, Epoch {epoch}: Average Train Loss: {avg_loss:.6f}, Train AUC: {auc_score:.4f}\n\n")
    return avg_loss

def val(model, device,data_loader, loss_criteria,batch_file_path, epoch, fold,Test=False):
    model.eval()
    test_loss = 0
    all_out_probs = []
    all_targets = []
    all_cls_tokens = []

    with open(batch_file_path, 'a') as f:
        f.write("Validating Progress\n")
        f.write(f"Fold {fold}, Epoch {epoch}\n")
        with torch.no_grad():
            for batch, tensor in enumerate(tqdm(data_loader, desc="Validating Progress")):
                data, target = tensor
                data = data.to(device)
                target = target.to(device)
                out, cls_token = model(data,None)
                out_prob = torch.sigmoid(out)
                loss = loss_criteria(out,target)
                test_loss += loss.item()
                out_prob_np = out_prob.cpu().detach().numpy()
                target_np = target.cpu().detach().numpy()
                if Test:
                    all_cls_tokens.append(cls_token.cpu().detach())
                all_out_probs.append(out_prob_np)
                all_targets.append(target_np)
                
                f.write(f"Batch {batch + 1}: Val Loss = {loss.item():.6f}\n")
    
        avg_loss = test_loss/(batch+1)
        if Test:
            all_cls_tokens = torch.cat(all_cls_tokens, dim=0)
        all_out_probs_np = np.concatenate(all_out_probs)
        all_targets_np = np.concatenate(all_targets)
        auc_score = calculate_masked_auc(all_targets_np, all_out_probs_np)
        print('Validation set: Average loss: {:.6f}, AUC: {:.4f}\n'.format(avg_loss, auc_score))
        f.write(f"Fold {fold}, Epoch {epoch}: Average Val Loss: {avg_loss:.6f}, Val AUC: {auc_score:.4f}\n\n")
    if Test:
        return avg_loss, auc_score,all_cls_tokens
    else:
        return avg_loss, auc_score