import torch
from fashionMNIST import NeuralNetwork

if __name__ == "__main__":
    model = NeuralNetwork()
    model = torch.load("pytorch-basics/models/model.pth", weights_only=False)
    model.eval()
    print(model)