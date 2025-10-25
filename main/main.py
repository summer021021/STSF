import os
import torch
import tempfile

from utils import *
from block import *
from dataset import *

from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR


def trainable(config):
    set_random_seed(config["seed"])
    epochs = config["epochs"]
    trainloader, testloader = MNISTLoader(config["batch_size"], config["T"])
    network = Net([config["inSize"]] + [config["layerSize"]] * config["nlayers"], nclass=config["nclass"],
                  threshold=config["threshold"], decay=config["decay"], vmax=config["vmax"])
    optimizer = SGD(network.parameters(), lr=config["lr"])
    scheduler = StepLR(optimizer, step_size=5, gamma=0.9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)

    for epoch in range(epochs):
        training_loss, training_accuracy = fit(network, trainloader, optimizer, device)
        testing_accuracy = test(network, testloader, device)
        if hasattr(network, 'global_loss_gradient'):
            print(f"Epoch {epoch}: global_loss_gradient = {network.global_loss_gradient}")
        scheduler.step()
        print(f"Epoch {epoch}: Training Loss = {training_loss:.4f}, Training Acc = {training_accuracy:.4f}, Test Acc = {testing_accuracy:.4f}")
    print("训练完成！")


def main():
    config = {
        "epochs": 100,
        "batch_size": 128,
        "lr": 5e-1,
        "T": 10,
        "threshold": 1.0,
        "decay": 1.0,
        "vmax": 1.0,
        "nclass": 10,
        "inSize": 784,
        "nlayers": 2,
        "layerSize": 800,
        "seed": 0,
    }
    trainable(config)


if __name__ == "__main__":
    main()
