import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data_loader import load_data
from model import FashionMNISTModel

GREEN = '\033[32m'
BOLD = '\033[1m'
RESET = '\033[0m'
BOLD_GREEN = '\033[1;32m'

def train_model(epochs=3, batch_size=64, lr=1e-3):
    train_data, _ = load_data()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    device = torch.device("cpu") 
    model = FashionMNISTModel().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, labels)
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running/len(train_loader):.4f}") 
    
    model_path = os.path.join(os.path.dirname(__file__), "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Modello salvato in {BOLD}{model_path}{RESET}")

if __name__ == "__main__":
    train_model()
