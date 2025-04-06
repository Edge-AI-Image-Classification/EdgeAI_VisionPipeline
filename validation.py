import torch

def evaluate(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0.0
    numCorrect = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (imgs, lbls) in enumerate(dataloader):
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, lbls)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            numCorrect += (preds == lbls).sum().item()
            total += lbls.size(0)
            if (batch_idx+1) % 10 == 0:
                print(f"Validation Batch {batch_idx+1}/{len(dataloader)} processed.")
            
    avg_loss = total_loss / len(dataloader)
    accuracy = numCorrect / total * 100
    return avg_loss, accuracy
