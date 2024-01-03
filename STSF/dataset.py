import scipy.io as sio

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


import torchvision
from torchvision.datasets import MNIST,FashionMNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda

import tonic
from tonic import DiskCachedDataset
import tonic.transforms as transforms

class Rate:
    def __init__(self, T):
        self.T = T

    def __call__(self, input):
        input = input.view(-1)
        output = torch.zeros((self.T, *input.shape))
        for t in range(self.T):
            output[t] = torch.rand_like(input).le(input).to(input)
        return output

class Real:
    def __init__(self, T):
        self.T = T

    def __call__(self, input):
        input = input.view(-1)
        output = torch.zeros((self.T, *input.shape))
        for t in range(self.T):
            output[t] = input.clone()
        return output

class Resize:
    def __init__(self, size):
        self.size = torchvision.transforms.Resize(size, antialias=True)

    def __call__(self, img):
        return torch.stack([self.size(torch.from_numpy(i)) for i in img], dim=0)
  
class NETtalk(Dataset):
    def __init__(self, path, train=True, transform=None, target_transform=None):
        super(NETtalk, self).__init__()
        mat_contents = sio.loadmat(path)
        if train:
            self.x_values = mat_contents['train_x']
            self.y_values = mat_contents['train_y']
        else:
            self.x_values = mat_contents['test_x']
            self.y_values = mat_contents['test_y']
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        sample = self.transform(self.x_values[index])
        label = self.target_transform(self.y_values[index])
        return sample, label

    def __len__(self):
        return len(self.x_values)

def MNISTLoader(batch_size, T):
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Rate(T),
        Lambda(lambda x: torch.flatten(x,start_dim = 1))
        ])
    trainloader = DataLoader(MNIST("/home/peace/code/Data", train=True, download=True, transform=transform), 
                             num_workers= 4, batch_size=batch_size, shuffle=True)

    testloader = DataLoader(MNIST("/home/peace/code/Data", train=False, download=True, transform=transform), 
                            num_workers= 4, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

def FasionMNISTLoader(batch_size, T):
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Rate(T),
        Lambda(lambda x: torch.flatten(x,start_dim = 1))
        ])
    trainloader = DataLoader(FashionMNIST("/home/peace/code/Data", train=True, download=True, transform=transform), 
                             num_workers= 4, batch_size=batch_size, shuffle=True)

    testloader = DataLoader(FashionMNIST("/home/peace/code/Data", train=False, download=True, transform=transform), 
                            num_workers= 4, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

def NETtalkTLoader(batch_size, T):
    transform = Compose([
        lambda x: torch.tensor(x, dtype=torch.float32),
        Real(T),
        ])
    target_transform = Compose([
        lambda x: torch.tensor(x, dtype=torch.int64),
        ])
    trainloader = DataLoader(NETtalk("/home/peace/code/Data/NETtalk/nettalk_small.mat",train=True, transform=transform, target_transform=target_transform),
                              batch_size=batch_size, shuffle=True)
    testloader = DataLoader(NETtalk("/home/peace/code/Data/NETtalk/nettalk_small.mat",train=False, transform=transform, target_transform=target_transform),
                              batch_size=batch_size, shuffle=True)
    return trainloader, testloader

def DVSGestureLoader(batch_size, T):
    transform = transforms.Compose([
        transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size, n_time_bins = T),
        Resize((32,32)),
        lambda x : x.reshape(x.shape[0], -1)
    ])

    trainloader = DataLoader(DiskCachedDataset(tonic.datasets.DVSGesture(save_to="/home/peace/code/DataSets", train=True, transform=transform), 
                                               cache_path=f"/home/peace/code/DataSets/DVSGesture/cache/{T}/train"), 
                            num_workers= 4, batch_size=batch_size, shuffle=True, collate_fn=tonic.collation.PadTensors(batch_first=True))
    testloader = DataLoader(DiskCachedDataset(tonic.datasets.DVSGesture(save_to="/home/peace/code/DataSets", train=False, transform=transform), 
                                               cache_path=f"/home/peace/code/DataSets/DVSGesture/cache/{T}/test"), 
                            num_workers= 4, batch_size=batch_size, shuffle=False, collate_fn=tonic.collation.PadTensors(batch_first=True))
    return trainloader, testloader

def NMNISTLoader(batch_size, T):
    transform = transforms.Compose([
        transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size, n_time_bins = T),
        lambda x : x.reshape(x.shape[0],-1)
        ])
    trainloader = DataLoader(DiskCachedDataset(tonic.datasets.NMNIST(save_to="/home/peace/code/DataSets", train=True, transform=transform), 
                                               cache_path=f"/home/peace/code/DataSets/NMNIST/cache/{T}/train"), 
                            num_workers= 4, batch_size=batch_size, shuffle=True, collate_fn=tonic.collation.PadTensors(batch_first=True))
    testloader = DataLoader(DiskCachedDataset(tonic.datasets.NMNIST(save_to="/home/peace/code/DataSets", train=False, transform=transform), 
                                               cache_path=f"/home/peace/code/DataSets/NMNIST/cache/{T}/test"), 
                            num_workers= 4, batch_size=batch_size, shuffle=False, collate_fn=tonic.collation.PadTensors(batch_first=True))
    return trainloader, testloader


