import os
import random

import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

from utils import default_device, constants
from utils.image_folder_with_path import ImageFolderWithPaths

torch.manual_seed(constants.RANDOM_SEED)

def to_device(data, device):
    """
    Move tensor(s) to chosen device
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """
    Wrap a DataLoader to move data to a device
    """
    def __init__(self, data_loader, device):
        self.data_loader = data_loader
        self.device = device
    
    def __iter__(self):
        """
        yield a batch of data after moving it to device
        """
        for b in self.data_loader:
            yield to_device(b, self.device)
    
    def __len__(self):
        """
        return number of batch size
        """
        return len(self.data_loader)

default_device = default_device.device

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=random.uniform(5,10)),
    transforms.Resize((512,512)),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
])

classes = os.listdir(constants.DATA_PATH + constants.TRAIN_PATH)
train_dataset = ImageFolder(constants.DATA_PATH + constants.TRAIN_PATH, transform=train_transforms)
valid_dataset = ImageFolder(constants.DATA_PATH + constants.VAL_PATH, transform=test_transforms)
test_dataset = ImageFolderWithPaths(constants.DATA_PATH + constants.TEST_PATH, transform=test_transforms)

train_dl = DataLoader(train_dataset, constants.BATCH_SIZE, shuffle=True, num_workers=constants.NUM_WORKER, pin_memory=True)
val_dl = DataLoader(valid_dataset, constants.BATCH_SIZE, num_workers=constants.NUM_WORKER, pin_memory=True)
test_dl = DataLoader(test_dataset, constants.BATCH_SIZE, num_workers=constants.NUM_WORKER, pin_memory=True)

# automatically transferring batches of data to GPU
train_dl = DeviceDataLoader(train_dl, default_device)
val_dl = DeviceDataLoader(val_dl, default_device)
test_dl = DeviceDataLoader(val_dl, default_device)

