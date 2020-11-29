
import torch
import torch.nn as nn
import torchvision
class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)
        kernelCount = self.densenet121.classifier.in_features
        self.densenet121.self_classifier = nn.Sequential(nn.Linear(kernelCount, 14))
    def gradients(self,outputs, inputs):
        a = torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),create_graph=True)
        return a
    def forward(self, x,flag=None):
        u = self.densenet121(x)
        return u

