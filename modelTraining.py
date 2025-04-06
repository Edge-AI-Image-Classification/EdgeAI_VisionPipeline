import torch
import torch.nn as nn
from torch.optim import SGD
from dataPreprocessLoader import trainDataloader, valDataloader
from resnet50 import ResNet50
from validation import evaluate


device = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50(num_classes=80)
model=model.to(device)

# during validation stage, modify these hyperparameters
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
# optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()
epochs = 1 # change this as we train the model


def trainModel(dataloader, model, loss_fn, optimizer, epochs):
    for epoch in range(epochs):
        print(f"\nStarting epoch {epoch+1}/{epochs}")
        model.train()
        lossVal = 0
        numCorrect = 0
        total = 0

        for batch_idx, (imgs, lbls) in enumerate(dataloader):
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, lbls)
            loss.backward()
            optimizer.step()
            lossVal += loss.item()
            preds = torch.argmax(out, dim=1)
            numCorrect += (preds == lbls).sum().item()
            total+= lbls.size(0)

            if (batch_idx+1)%10==0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}: Loss = {loss.item():.4f}")
        
        avg_loss = lossVal / len(dataloader)
        accuracy = numCorrect / total * 100
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Run validation after each epoch
        val_loss, val_acc = evaluate(valDataloader, model, loss_fn, device)
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

        print(f"Epoch {epoch+1} - Loss: {lossVal/len(dataloader):.4f}")
        accuracy = numCorrect / total * 100
        print(f"Epoch {epoch+1} - Accuracy: {accuracy:.4f}")

trainModel(trainDataloader, model, loss_fn, optimizer,epochs)

#saving the trained model to run inference on edge device
torch.save(model.state_dict(), "resnet_coco.pth")
print("Model saved to resnet_coco.pth")

