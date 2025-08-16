from torchvision import datasets, transforms

def load_data(data_dir="data"):
    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_data = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    return train_data, test_data
