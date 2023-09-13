import torch.nn as nn
from torchvision import models
import numpy as np
from torchvision import transforms

MEAN = [0.46215433, 0.46217325, 0.46218519]
STD = [0.24857047, 0.24856126, 0.24856159]

transform_features = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Normalize(MEAN, STD)
])

class ExtractFeatures(nn.Module):
    def __init__(self):
        super(ExtractFeatures, self).__init__()
        model = models.vgg16()
        self.features = nn.Sequential(model.features, model.avgpool)
        self.linear = model.classifier[0]

    def forward(self, input):
        x = self.features(input)
        x = x.view(1, -1)
        x = self.linear(x)
        return x
    
