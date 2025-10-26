import torch
import torch.nn as nn
from utils import SFMatrix


quantize_range = 1.0


def WeightQuantize(weight):
    """
    Quantize weights to 8-bit integers.
    
    将权重量化为8位整数。
    """
    print(weight.abs().max())
    weight = weight * quantize_range
    # weight = torch.round(weight)
    # weight = torch.clamp(weight, -quantize_range, quantize_range)
    return weight


class HardwareLIF(nn.Module):
    """
    Hardware LIF neuron model with a simple reset mechanism.
    
    模拟一个硬件LIF神经元模型，具有简单的重置机制。
    对比原模型，固定采用硬重置方式，阈值和衰减率为1.0，最大电压为1.0。
    神经元激活函数集成在前向传播中，使用torch的张量操作来模拟神经元的行为。
    """
    def __init__(self):
        super().__init__()
        self.spikes = None
        self.v = None

    def forward(self, x):
        if self.spikes is None:
            self.spikes = torch.zeros_like(x)
            self.v = torch.zeros_like(x)
        else:
            self.v.detach_()
            self.spikes.detach_()

        # 硬重置方式
        self.v = self.v * (1 - self.spikes) + x
        self.spikes = (self.v > quantize_range).float()

        return self.spikes
    
    def reset(self):
        self.spikes = None
        self.v = None


class HardwareBlock(nn.Module):
    """
    Hardware Block that combines a synapse and a neuron.
    
    硬件块，结合了突触和神经元。
    该块使用一个线性层作为突触，并使用硬件LIF神经元进行处理。
    """
    def __init__(self, in_features, out_features, n_class):
        super().__init__()
        self.synapse = nn.Linear(in_features, out_features, bias=False)
        self.neuron = HardwareLIF()
        self.classifier = HardwareClassifier(out_features, n_class)

        self.synapse.weight.data = nn.Parameter(WeightQuantize(self.synapse.weight.data))
        self.classifier.synapse.weight.data = nn.Parameter(WeightQuantize(self.classifier.synapse.weight.data))

    def forward(self, input):
        deltaV = self.synapse(input)
        spike = self.neuron(deltaV)
        return spike
    
    def fit(self, input, labels, lr, loss):
        spike = self.forward(input)
        if self.training:
            output = self.classifier(spike)  # [batch_size, n_class]

            # 仅用于统计损失
            loss += nn.MSELoss()(output, labels)

            # 1. 计算 classifier 输出梯度
            grad_output = 2 * (output - labels) / output.shape[1]  # [batch_size, n_class]

            # 2. 计算 classifier DFA梯度
            grad_dfa = torch.matmul(grad_output, self.classifier.Fa)  # [batch_size, out_features]

            # 3. 代理梯度：损失对 deltaV 的梯度
            grad_deltaV = grad_dfa * spike  # 这里 spike 作为代理梯度  # [batch_size, out_features]

            # 4. 用输入和 grad_deltaV 计算得到 synapse 权重变化量
            grad_w = torch.matmul(grad_deltaV.t(), input) / input.shape[0]  # [out_features, input_dim]

            # 5. 更新 synapse 权重
            self.synapse.weight.data -= lr * grad_w * quantize_range

            # 6. 计算 classifier 权重变化量
            grad_w = torch.matmul(grad_output.t(), spike) / spike.shape[0]

            # 7. 更新 classifier 权重
            self.classifier.synapse.weight.data -= lr * grad_w * quantize_range

        return spike.detach(), loss, output.detach()
    
    def reset(self):
        self.neuron.reset()
        self.classifier.reset()


class HardwareClassifier(nn.Module):
    """
    Hardware Classifier for the final layer of the network.
    
    硬件分类器，用于网络的最后一层。
    该分类器使用一个简单的线性层来进行分类。
    """
    def __init__(self, in_features, n_class):
        super().__init__()
        self.in_features = in_features
        self.n_class = n_class
        self.synapse = nn.Linear(in_features, n_class, bias=False)
        self.neuron = HardwareLIF()  
        self.Fa = nn.Parameter(torch.Tensor(SFMatrix((in_features, n_class))), requires_grad=False)

    def forward(self, input):
        deltaV = self.synapse(input)
        spike = self.neuron(deltaV)
        return spike
    
    def reset(self):
        self.neuron.reset()


class HardwareNet(nn.Module):
    def __init__(self, dims, nclass):
        super().__init__()

        self.nclass = nclass
        self.modlist = []
        for d in range(len(dims) - 1):
            self.modlist.append(HardwareBlock(dims[d], dims[d+1], nclass))
        self.modlist = nn.ModuleList(self.modlist)
    
    def forward(self, input):
        input
        for modle in self.modlist:
            input = modle(input)
        with torch.no_grad():
            input = self.modlist[-1].classifier(input)
        return input
    
    def fit(self, input, labels, lr):
        loss = torch.zeros(1, device=input.device)
        labels = torch.nn.functional.one_hot(labels, self.nclass).float()
        for modle in self.modlist:
            input, loss, output = modle.fit(input, labels, lr, loss)
        return output, loss

    def reset(self):
        for modle in self.modlist:
            modle.reset()
