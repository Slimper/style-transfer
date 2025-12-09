import torch
import torch.nn as nn
import torchvision.models as models

class VGGFeatures(nn.Module):
    def __init__(self, layers=None):
        super(VGGFeatures, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.selected_layers = layers or ['0', '5', '10', '19', '28']
        self.vgg = vgg

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.selected_layers:
                features.append(x)
        return features
