from enum import Enum

class ModelType(Enum):
    NFNET = "nfnet"
    CUSTOM_CNN_CIFAR = "custom_cnn_cifar"
    CUSTOM_CNN_IMGNET = "custom_cnn_imgnet"