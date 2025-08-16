from setuptools import setup, find_packages
from fashion_mnist.model import FashionMNISTModel
from fashion_mnist.data_loader import load_data

setup(
    name="Fashion-MNIST-ML",
    version="0.1.0",
    description="CNN per Fashion-MNIST con Docker e CI",
    author="Domenico Pullia",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "torchvision",
        "matplotlib",
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "fashion-mnist-train=fashion_mnist.train:train_model",
            "fashion-mnist-eval=fashion_mnist.evaluate:evaluate_model",
        ],
    },
)
