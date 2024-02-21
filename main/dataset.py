import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda


class Rate:
    def __init__(self, T):
        self.T = T

    def __call__(self, input):
        input = input.view(-1)
        output = torch.zeros((self.T, *input.shape))
        for t in range(self.T):
            output[t] = torch.rand_like(input).le(input).to(input)
        return output

def MNISTLoader(batch_size, T):
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Rate(T),
        Lambda(lambda x: torch.flatten(x,start_dim = 1))
        ])
    trainloader = DataLoader(MNIST("../Data", train=True, download=True, transform=transform), 
                             num_workers= 4, batch_size=batch_size, shuffle=True)

    testloader = DataLoader(MNIST("../Data", train=False, download=True, transform=transform), 
                            num_workers= 4, batch_size=batch_size, shuffle=False)
    return trainloader, testloader
