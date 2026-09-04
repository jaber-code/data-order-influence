from model.simple_cnn import SimpleCNN
from model.simple_cnn_cifar import SimpleCNNCifar
from model.nf_net import NFNet
from model.model_type import ModelType

class ModelFactory:
    @staticmethod
    def create_model(model_type, num_classes=1000):
        if model_type == ModelType.CUSTOM_CNN_IMGNET:
            return SimpleCNN()
        elif model_type == ModelType.CUSTOM_CNN_CIFAR:
            return SimpleCNNCifar()
        elif model_type == ModelType.NFNET:
            return NFNet(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
