from tqdm import tqdm
import torch
import numpy as np
from lifelines.utils import concordance_index 
import pandas as pd

## train,val
def train(model, device, data_loader, optimizer,loss_criteria, batch_file_path, epoch, fold):
    model.train()
    train_loss = 0
    loss_logits = []
    loss_time = []
    loss_event = []
    cindex_logits = []
    cindex_time = []
    cindex_event = []

    with open(batch_file_path, 'a') as f:
        f.write("Training Progress:\n")
        f.write(f"Fold {fold}, Epoch {epoch}\n")
        for batch, tensor in enumerate(tqdm(data_loader, desc="Training Progress")):
            tensor = [x.to(device) if hasattr(x, "to") else x for x in tensor]
            *inputs, target = tensor
            optimizer.zero_grad()
            out, *_ = model(*inputs)
            loss_logits.append(out.squeeze(1))
            loss_time.append(target[:,1])
            loss_event.append(target[:,0])
            loss_logits = torch.stack(loss_logits).to(device)
            loss_time = loss_time[0].clone().detach().to(device)
            loss_event = loss_event[0].clone().detach().to(device)
            loss = loss_criteria(loss_logits, loss_time, loss_event)
            if len(_) ==5:
                aux_loss = 0.3 * (
                    loss_criteria(_[0], loss_time, loss_event) +
                    loss_criteria(_[1], loss_time, loss_event) +
                    loss_criteria(_[2], loss_time, loss_event) +
                    0.1 * loss_criteria(_[3], loss_time, loss_event)
                )
                loss = loss + aux_loss
            if len(_) ==4:
                aux_loss = 0.3 * (
                    loss_criteria(_[0], loss_time, loss_event) +
                    loss_criteria(_[1], loss_time, loss_event) +
                    loss_criteria(_[2], loss_time, loss_event)
                )
                loss = loss + aux_loss
            train_loss += loss.item()

            loss_logits = []
            loss_time = []
            loss_event = []

            cindex_logits.append(out.squeeze(1).cpu().detach().numpy())
            cindex_time.append(target[:,1].cpu().detach().numpy())
            cindex_event.append(target[:,0].cpu().detach().numpy()) 

            loss.backward()
            optimizer.step()
            f.write(f"Batch {batch + 1}: Train Loss = {loss.item():.6f}\n")
        avg_loss = train_loss / (batch+1)
        cindex_logits_np = np.concatenate(cindex_logits)
        cindex_time_np = np.concatenate(cindex_time)
        cindex_event_np = np.concatenate(cindex_event)
        cindex = concordance_index(cindex_time_np,-np.exp(cindex_logits_np),cindex_event_np)
        print('Training set: Average loss: {:.6f}, Cindex: {:.4f}\n'.format(avg_loss, cindex))
        f.write(f"Fold {fold}, Epoch {epoch}: Average Train Loss: {avg_loss:.6f}, Train Cindex: {cindex:.4f}\n\n")
    return avg_loss

def val(model, device,data_loader, loss_criteria,batch_file_path, epoch, fold,Test=False):
    model.eval()
    test_loss = 0
    loss_logits = []
    loss_time = []
    loss_event = []
    cindex_logits = []
    cindex_time = []
    cindex_event = []

    with open(batch_file_path, 'a') as f:
        f.write("Validating Progress\n")
        f.write(f"Fold {fold}, Epoch {epoch}\n")
        with torch.no_grad():
            for batch, tensor in enumerate(tqdm(data_loader, desc="Validating Progress")):
                tensor = [x.to(device) if hasattr(x, "to") else x for x in tensor]
                *inputs, target = tensor
                out, *_ = model(*inputs)
                loss_logits.append(out.squeeze(1))
                loss_time.append(target[:,1])
                loss_event.append(target[:,0])
                loss_logits = torch.stack(loss_logits).to(device)
                loss_time = loss_time[0].clone().detach().to(device)
                loss_event = loss_event[0].clone().detach().to(device)
                loss = loss_criteria(loss_logits, loss_time, loss_event)
                test_loss += loss.item()

                loss_logits = []
                loss_time = []
                loss_event = []

                cindex_logits.append(out.squeeze(1).cpu().detach().numpy())
                cindex_time.append(target[:,1].cpu().detach().numpy())
                cindex_event.append(target[:,0].cpu().detach().numpy()) 

                f.write(f"Batch {batch + 1}: Val Loss = {loss.item():.6f}\n")
    
        avg_loss = test_loss/(batch+1)
        cindex_logits_np = np.concatenate(cindex_logits)
        cindex_time_np = np.concatenate(cindex_time)
        cindex_event_np = np.concatenate(cindex_event)
        prob_saved = pd.DataFrame({
        'Logits': cindex_logits_np,
        'target_y': cindex_event_np,
        'BL2Target_yrs': cindex_time_np
        })
        cindex = concordance_index(cindex_time_np,-np.exp(cindex_logits_np),cindex_event_np)
        print('Validation set: Average loss: {:.6f}, Cindex: {:.4f}\n'.format(avg_loss, cindex))
        f.write(f"Fold {fold}, Epoch {epoch}: Average Val Loss: {avg_loss:.6f}, Val Cindex: {cindex:.4f}\n\n")
    return avg_loss,cindex,prob_saved