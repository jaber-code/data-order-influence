import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from dataset.dataset_loader import DatasetLoader

class ImageNetLoader(DatasetLoader):

    PATH_IMAGENET_TRAINING = '/fscratch/sjaber/imagenet/train/' #/ds/images/imagenet
    PATH_IMAGENET_TESTING = '/fscratch/sjaber/imagenet/val_folders/'

    PATH_IMAGENET_CLASSES_NAMES = '/ds/images/imagenet/imagenet_metadata.txt'

    def __init__(self, dataset_path, batch_size):
        self.dataset_path = dataset_path
        self.training_path = dataset_path + "/train/"
        self.validation_path = dataset_path + "/val_folders/"
        self.batch_size = batch_size


    def load_data(self):
        print("Loading Imagenet from a folder..")

        training_transform_1 = transforms.Compose([
            transforms.RandomResizedCrop(192),
            transforms.RandomHorizontalFlip(p=0.55),  
            transforms.RandomRotation(16),
            transforms.ColorJitter(brightness=0.29, contrast=0.29, saturation=0.29, hue=0.12), 
            transforms.RandomAffine(degrees=15, translate=(0.07, 0.07), scale=(0.9, 1.1)), 
            transforms.ToTensor(),  
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.05), ratio=(0.3, 2.0), value='random'),         
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ]) 


        training_transform_2 = transforms.Compose([
            transforms.RandomResizedCrop(192),
            transforms.RandomHorizontalFlip(p=0.55),  
            transforms.RandomRotation(16),
            transforms.ColorJitter(brightness=0.29, contrast=0.29, saturation=0.29, hue=0.12), 
            transforms.RandomAffine(degrees=15, translate=(0.07, 0.07), scale=(0.9, 1.1)), 
            transforms.ToTensor(),  
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.05), ratio=(0.3, 2.0), value='random'),         
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ]) 


        train_dataset = torchvision.datasets.ImageFolder(root=self.training_path, transform=training_transform_1)

        if torch.cuda.is_available()  and torch.distributed.is_initialized():
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
        else:
            train_sampler = None

        num_samples_per_gpu = len(train_dataset) // torch.distributed.get_world_size()
        print(f"Number of samples per GPU: {num_samples_per_gpu}")

        trainloader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
        classes_ids = train_dataset.classes

        classes_names = []
        with open(self.PATH_IMAGENET_CLASSES_NAMES, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) > 1 and parts[0] in classes_ids:
                    classes_names.append(parts[1])


        print("classes names", classes_names[:10])
        
        val_transform = transforms.Compose([
            transforms.Resize(288),
            transforms.CenterCrop(256),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_dataset = torchvision.datasets.ImageFolder(root=self.validation_path, transform=val_transform)
        testloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)

        print(f"Classes size: {len(classes_names)}")
        print(f"Sample class: {classes_names[0]}")
        print("Finished Loading")

        total_samples = len(train_dataset)
        samples_per_gpu = (total_samples + torch.distributed.get_world_size() - 1) // torch.distributed.get_world_size()  # Rounded up
        total_padded_samples = samples_per_gpu * torch.distributed.get_world_size()

        print(f"Total samples: {total_samples}")
        print(f"Samples per GPU: {samples_per_gpu}")
        print(f"Total padded samples (including padding): {total_padded_samples}")

        return train_dataset, trainloader, testloader, classes_names, train_sampler