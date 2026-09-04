from dataset.dataset_type import DataSetType
from imagenet_loader import ImageNetLoader
from dataset.imagenet_tar_loader import ImageNetTarLoader
from dataset.cifar_10_loader import Cifar10Loader

class DatasetHandler:
    def __init__(self, dataset_type, dataset_path , batch_size, is_local=False):
        self.dataset_type = dataset_type
        self.is_local = is_local
        self.dataset_path = dataset_path
        self.batch_size = batch_size

    def load_data_set(self):
        loader = self._get_loader()
        return loader.load_data()

    def _get_loader(self):
        if self.dataset_type == DataSetType.CIFAR10:
            return Cifar10Loader(is_local=self.is_local)
        elif self.dataset_type == DataSetType.IMAGE_NET_1K_TAR:
            return ImageNetTarLoader()
        elif self.dataset_type == DataSetType.IMAGE_NET_1K:
            return ImageNetLoader(self.dataset_path, self.batch_size)
        else:
            raise RuntimeError("Unhandled dataset type!")
