import os
import torch
import torch

from hardware_utlis import *
from hardware_block import *
from dataset import *

import matplotlib.pyplot as plt


def trainable(config):
    set_random_seed(config["seed"])
    epochs = config["epochs"]
    initial_lr = config["lr"]
    decay_step = 10
    decay_rate = 0.5

    trainloader, testloader = MNISTLoader(config["batch_size"], config["T"])
    network = HardwareNet([config["inSize"]] + [config["layerSize"]] * config["nlayers"], nclass=config["nclass"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)

    best_acc = 0.0
    best_model = None
    training_loss_record = []
    training_accuracy_record = []
    testing_accuracy_record = []
    for epoch in range(epochs):
        lr = initial_lr * (decay_rate ** (epoch // decay_step))
        training_loss, training_accuracy = fit(network, trainloader, lr, device)
        testing_accuracy = test(network, testloader, device)
        print(f"Epoch {epoch}: Training Loss = {training_loss:.4f}, Training Acc = {training_accuracy:.4f}, Test Acc = {testing_accuracy:.4f}")

        training_loss_record.append(training_loss)
        training_accuracy_record.append(training_accuracy)

        # Save the best model
        if testing_accuracy > best_acc:
            best_acc = testing_accuracy
            best_model = network.state_dict()
        testing_accuracy_record.append(testing_accuracy)
    print("训练完成！")

    plt.figure()
    plt.title("Hardware Network Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(epochs), training_loss_record, label="Training Loss")
    plt.legend()
    plt.savefig("hardware_network_loss_curve.png", dpi=300)
    
    plt.figure()
    plt.title("Hardware Network Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(range(epochs), training_accuracy_record, label="Training Accuracy")
    plt.plot(range(epochs), testing_accuracy_record, label="Testing Accuracy")
    plt.legend()
    plt.savefig("hardware_network_accuracy_curve.png", dpi=300)

    print(f"Best Test Accuracy: {best_acc:.4f}")
    # Save the best model state_dict
    torch.save(best_model, "best_hardware_model.pth")


def main():
    config = {
        "epochs": 100,
        "batch_size": 128,
        "lr": 1,
        "T": 8,
        "nclass": 10,
        "inSize": 784,
        "nlayers": 2,
        "layerSize": 1024,
        "seed": 0,
    }
    trainable(config)

if __name__ == "__main__":
    main()