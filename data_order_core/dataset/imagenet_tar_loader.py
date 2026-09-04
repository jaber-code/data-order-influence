import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset.dataset_loader import DatasetLoader
from dataset.tar_imagenet_dataset import TarImageNetDataset

class ImageNetTarLoader(DatasetLoader):

    PATH_IMAGENET_TRAINING_TAR = '/ds/images/ImageNet_2022/train/ILSVRC2012_img_train.tar'
    PATH_IMAGENET_TESTING_TAR = '/ds/images/ImageNet_2022/val/ILSVRC2012_img_val.tar'

    def load_data(self):
        print("Loading Imagenet from a tar file..")
        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        train_dataset = TarImageNetDataset(self.PATH_IMAGENET_TRAINING_TAR, transform=transform)
        trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)

        test_dataset = TarImageNetDataset(self.PATH_IMAGENET_TESTING_TAR, transform=transform)
        testloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

        class_names = train_dataset.get_class_names()

        print("Classes size:", len(class_names))
        print("Sample class:", class_names[0])
        print("Finished Loading")
        return train_dataset, trainloader, testloader, class_names