import math
import torch
import numpy as np


def fit(model, train_loader, lr, device):
    model.train()

    train_loss, train_acc,n = 0, 0, 0
    for data, target in train_loader:
        data, target = data.transpose(0, 1).to(device), target.to(device)
        model.reset()
        output = None
        for i in range(data.shape[0]):
            out, loss = model.fit(data[i], target, lr)

            if output is None:
                output = out.clone()
            else:
                output += out.clone()

            train_loss += loss.item()

        n += target.shape[0]
        pred = output.argmax(dim=1,keepdim=True)
        train_acc += pred.eq(target.view_as(pred)).sum().item()
        
    return train_loss / n, train_acc / n

def test(model, test_loader, device):
    model.eval()

    test_acc,n = 0, 0
    for data, target in test_loader:
        data, target = data.transpose(0, 1).to(device), target.to(device)
        model.reset()
        output = None
        for i in range(data.shape[0]):
            out = model(data[i])

            if output is None:
                output = out.clone()
            else:
                output += out.clone()

        n += target.shape[0]
        pred = output.argmax(dim=1, keepdim=True)
        test_acc += pred.eq(target.view_as(pred)).sum().item()
    return test_acc / n

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def SFMatrix(size: tuple):
    input_size, output_size = size
    # get SF mask
    perm = torch.randperm(math.ceil((input_size // output_size) + 1) * output_size)
    index = (perm % output_size)[:input_size]
    mask = torch.zeros(size).scatter_(1, index.unsqueeze(1), 1)
    # init
    bd = np.sqrt(output_size / input_size)
    matrix = (2 * bd * torch.rand(size) - bd) * mask
    return matrix.T
