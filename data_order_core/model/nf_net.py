import timm
import torch.nn as nn
from model.base_model import BaseModel

class NFNet(BaseModel):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.nfnet = timm.create_model('nfnet_f0', pretrained=False)
        self.nfnet.fc = nn.Linear(self.nfnet.num_features, num_classes)

    def forward(self, x):
        return self.nfnet(x)
