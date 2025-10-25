import torch
import torch.nn as nn
from utils import SFMatrix
from torch.autograd import Function,grad
import csv


class ActFun(Function):
    @staticmethod
    def forward(ctx, input, islast, vmax=1.0):
        spikes = (input>0.)
        ctx.save_for_backward(input, spikes)
        ctx.islast = islast
        ctx.vmax = vmax
        return spikes.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, spikes = ctx.saved_tensors
        grad_input = grad_output.clone()
        if not ctx.islast:
            grad_input = grad_input * spikes
        return grad_input, None, None


class LIF(nn.Module):

    def __init__(self, threshold:float=1., decay:float=1., hardReset:bool=True, islast = False, vmax:float = 1.):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.vmax = vmax

        self.hardReset = hardReset
        self.islast = islast

        self.spikes = None
        self.v = None

    def forward(self, x):
        if self.spikes is None:
            self.spikes = torch.zeros_like(x)
            self.v = torch.zeros_like(x)
        else:
            self.v.detach_()
            self.spikes.detach_()

        if self.hardReset:
            self.v = self.v * (1 - self.spikes) * self.decay + x
        else:
            self.v = (self.v - self.spikes * self.threshold) * self.decay + x

        self.spikes = ActFun.apply(self.v - self.threshold, self.islast, self.vmax)

        return self.spikes
    
    def reset(self):
        self.v = None
        self.spikes = None


class Block(nn.Module):
    def __init__(self, in_features, out_features, n_class, threshold=1., decay=1., vmax=1.):
        super(Block, self).__init__()
            
        self.out_features = out_features
        self.synapse = nn.Linear(in_features, out_features, bias=False)
        self.neuron = LIF(threshold=threshold, decay=decay, vmax=vmax)

        self.classifier = Classifier(out_features, n_class, threshold=threshold, decay=decay, vmax=vmax)
        self.lossfun = nn.MSELoss()
        
    def forward(self, input):
        deltaV = self.synapse(input)
        spike = self.neuron(deltaV)
        return spike

    def fit(self, input, labels, loss):
        spike = self.forward(input)
        if self.training:
            output = self.classifier(spike)
            loss += self.lossfun(output, labels)
            g = grad(loss, output, retain_graph=True)[0]
            self.classifier.SetGlobalLoss(g)
        return spike.detach(), loss

    def reset(self):
        self.neuron.reset()
        self.classifier.reset()


class Classifier(nn.Module):
    def __init__(self, in_features, n_class, threshold=1., decay=1., vmax=1.):
        super().__init__()
        self.in_features = in_features
        self.n_class = n_class
        self.synapse = nn.Linear(in_features, n_class, bias=False)
        self.neuron = LIF(threshold=threshold, decay=decay, vmax=vmax, islast=True)
        self.Fa = nn.Parameter(torch.Tensor(SFMatrix((in_features, n_class))), requires_grad=False)

        self.register_full_backward_hook(self.dfa_backward_hook) 

    def forward(self, input):
        deltaV = self.synapse(input)
        spike = self.neuron(deltaV)
        return spike
    
    def SetGlobalLoss(self, global_loss_gradient):
        self.global_loss_gradient = global_loss_gradient
    
    @staticmethod
    def dfa_backward_hook(module, grad_input, grad_output):
        if grad_input[0] is None:
            return grad_input
        else:
            grad_dfa = torch.mm(module.global_loss_gradient, module.Fa)
            return (grad_dfa, *grad_input[1:-1])
    
    def reset(self):
        self.neuron.reset()


class Net(nn.Module):
    def __init__(self, dims, nclass, threshold=1., decay=1., vmax=1.):
        super().__init__()

        self.nclass = nclass
        self.modlist = []
        for d in range(len(dims) - 1):
            self.modlist.append(Block(dims[d], dims[d+1], nclass, threshold=threshold, decay=decay, vmax=vmax))
        self.modlist = nn.ModuleList(self.modlist)
    
    def forward(self, input):
        input
        for modle in self.modlist:
            input = modle(input)
        with torch.no_grad():
            input = self.modlist[-1].classifier(input)
        return input
    
    def fit(self, input, labels):
        loss = torch.zeros(1, device=input.device)
        labels = torch.nn.functional.one_hot(labels, self.nclass).float()
        for modle in self.modlist:
            input, loss = modle.fit(input, labels, loss)
        with torch.no_grad():
            input = self.modlist[-1].classifier(input)
        return input, loss

    def reset(self):
        for modle in self.modlist:
            modle.reset()
