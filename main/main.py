import os
import torch
import tempfile

from utils import *
from block import *
from dataset import *

from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt


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

    training_loss_record = []
    training_accuracy_record = []
    testing_accuracy_record = []
    for epoch in range(epochs):
        training_loss, training_accuracy = fit(network, trainloader, optimizer, device)
        testing_accuracy = test(network, testloader, device)
        if hasattr(network, 'global_loss_gradient'):
            print(f"Epoch {epoch}: global_loss_gradient = {network.global_loss_gradient}")
        scheduler.step()
        print(f"Epoch {epoch}: Training Loss = {training_loss:.4f}, Training Acc = {training_accuracy:.4f}, Test Acc = {testing_accuracy:.4f}")

        training_loss_record.append(training_loss)
        training_accuracy_record.append(training_accuracy)
        testing_accuracy_record.append(testing_accuracy)
    print("训练完成！")


    plt.figure()
    plt.title("Network Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(epochs), training_loss_record, label="Training Loss")
    plt.legend()
    plt.savefig("network_loss_curve.png", dpi=300)
    
    plt.figure()
    plt.title("Network Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(range(epochs), training_accuracy_record, label="Training Accuracy")
    plt.plot(range(epochs), testing_accuracy_record, label="Testing Accuracy")
    plt.legend()
    plt.savefig("network_accuracy_curve.png", dpi=300)


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
