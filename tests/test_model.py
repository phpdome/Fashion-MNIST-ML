import torch
from src.model import FashionMNISTModel

def test_forward_shape():
    model = FashionMNISTModel()
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    assert tuple(out.shape) == (2, 10)
