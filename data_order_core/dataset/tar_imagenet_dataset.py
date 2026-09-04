import os
import tarfile
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


class TarImageNetDataset(Dataset):
    def __init__(self, tar_path, transform=None):
        self.tar_path = tar_path
        self.transform = transform
        self.image_list = []
        self.labels = []

        with tarfile.open(tar_path, 'r') as main_tar:
            # Extract and list all class tar files (e.g., n01440764.tar)
            class_tar_files = [member for member in main_tar.getmembers() if member.isfile()]
            
            for class_tar_info in class_tar_files:
                class_tar = main_tar.extractfile(class_tar_info)
                with tarfile.open(fileobj=class_tar, mode='r') as class_tar:
                    for member in class_tar.getmembers():
                        if member.isfile() and member.name.endswith(('.jpg', '.jpeg', '.png')):
                            self.image_list.append((class_tar_info.name, member.name))
                            self.labels.append(os.path.basename(class_tar_info.name).split('.')[0])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        main_tar_name, image_name = self.image_list[idx]
        label = self.labels[idx]

        with tarfile.open(self.tar_path, 'r') as main_tar:
            class_tar_info = main_tar.getmember(main_tar_name)
            class_tar = main_tar.extractfile(class_tar_info)
            with tarfile.open(fileobj=class_tar, mode='r') as class_tar:
                img_file = class_tar.extractfile(image_name)

                try:
                    image = Image.open(img_file).convert('RGB')
                except UnidentifiedImageError:
                    print(f"Failed to load image: {image_name} from class tar: {main_tar_name}")
                    return None

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label
    
    def get_class_names(self):
        
        return [img_info.name for img_info in self.image_list]
