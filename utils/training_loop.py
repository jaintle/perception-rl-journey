"""
Reusable PyTorch training utilities.
Will grow over the 6-month journey.
"""

import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for xb, yb in tqdm(dataloader):
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            total_loss += loss.item()
            correct += (preds.argmax(dim=1) == yb).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)
