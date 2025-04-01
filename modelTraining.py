from dataPreprocessLoader import trainDataloader
import torch.nn as nn
from torch.optim import SGD, Adam
import torch

# WAITING ON MODEL SO FUNCTION IS UNTESTED ATM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()
epochs = 1 # change this as we train the model


def trainModel(dataloader, model, optimizer, loss_fn, epochs):
    for epoch in range(epochs):
        model.train()
        lossVal = 0

        for imgs, lbls in dataloader:
            imgs.to(device)
            lbls.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, lbls)
            loss.backward()
            optimizer.step()
            lossVal += loss.item()
            preds = torch.argmax(preds, dim=1)


        print(f"Epoch {epoch+1} - Loss: {lossVal/len(dataloader):.4f}")


trainModel(trainDataloader, model,loss_fn,optimizer,epochs)

