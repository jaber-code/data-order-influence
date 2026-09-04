import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset.dataset_loader import DatasetLoader


class Cifar10Loader(DatasetLoader):

    PATH_CIFAR = '/netscratch/sjaber/data'
    PATH_CIFAR_LOCAL = 'data'

    def __init__(self, is_local=False):
        self.is_local = is_local

    def load_data(self):
        print("Loading CIFAR10..")

        path_to_file = self.PATH_CIFAR
        if self.is_local:
            path_to_file = self.PATH_CIFAR_LOCAL
        
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = torchvision.datasets.CIFAR10(root=path_to_file, train=True,
                                                download=True, transform=transform)
        trainloader = DataLoader(train_dataset, batch_size=32,
                                shuffle=True, num_workers=2)

        test_dataset = torchvision.datasets.CIFAR10(root=path_to_file, train=False,
                                            download=True, transform=transform)
        testloader = DataLoader(test_dataset, batch_size=32,
                                shuffle=False, num_workers=2)

        print("Finished Loading")
        return train_dataset, trainloader, testloader, train_dataset.classes, None 