import os
import sys
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_loader import load_data
from model import FashionMNISTModel
GREEN = '\033[32m'
BOLD = '\033[1m'
RESET = '\033[0m'
BOLD_GREEN = '\033[1;32m'

LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def evaluate_model(batch_size=64):
    print("Avvio valutazione del modello...")
    _, test_data = load_data()
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    device = torch.device("cpu")
    model = FashionMNISTModel().to(device)

    # carica SEMPRE dallo stesso path dello script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.pth")
    if not os.path.exists(model_path):
        print(f"ERRORE: modello non trovato in {model_path}")
        print("Esegui prima il training (MODE=train o MODE=both).")
        sys.exit(1)

    # usa weights_only=True per evitare warning e caricare solo i pesi
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    acc = 100 * correct / total
    print(f"{BOLD}ACCURACY:{RESET} {BOLD_GREEN}{acc:.2f}%{RESET}")

    imgs, labels = next(iter(test_loader))
    outputs = model(imgs.to(device))
    _, pred = torch.max(outputs, 1)

    # Mostra 5 immagini con relative etichette
    fig = plt.figure(figsize=(10, 4))
    for i in range(5):
        ax = fig.add_subplot(1, 5, i+1)
        ax.imshow(imgs[i].squeeze(), cmap="gray")
        ax.set_title(LABELS[pred[i]])
        ax.axis("off")
    plt.show()

if __name__ == "__main__":
    evaluate_model()
