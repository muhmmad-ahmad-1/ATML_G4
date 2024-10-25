import torch
import torch.nn as nn
from torchvision.models import vgg11, VGG11_Weights, vgg16, VGG16_Weights, vgg19, VGG19_Weights
from torchsummary import summary

class VGG11(nn.Module):
    def __init__(self,n_classes,teacher=False,device="cuda"):
        super(VGG11,self).__init__()
        if teacher:
            self.model = vgg11(weights=VGG11_Weights,progress=True)
            for params in self.model.parameters():
                params.requires_grad = False
        else:
            self.model = vgg11()
        if n_classes != 1000:
            self.model.classifier[6] = nn.Linear(4096,n_classes,device=device)
    
    def forward(self,x):
        return self.model(x)

class VGG16(nn.Module):
    def __init__(self,n_classes,teacher=True,device="cuda"):
        super(VGG16,self).__init__()
        if teacher:
            self.model = vgg16(weights=VGG16_Weights,progress=True)
            for params in self.model.parameters():
                params.requires_grad = False
        else:
            self.model = vgg16()
        if n_classes != 1000:
            self.model.classifier[6] = nn.Linear(4096,n_classes,device=device)
    
    def forward(self,x):
        return self.model(x)

class VGG19(nn.Module):
    def __init__(self,n_classes,teacher=False,device="cuda"):
        super(VGG19,self).__init__()
        if teacher:
            self.model = vgg19(weights=VGG19_Weights,progress=True)
            for params in self.model.parameters():
                params.requires_grad = False
        else:
            self.model = vgg19()
        if n_classes != 1000:
            self.model.classifier[6] = nn.Linear(4096,n_classes,device=device)
    
    def forward(self,x):
        return self.model(x)
    
    