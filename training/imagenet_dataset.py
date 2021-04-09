import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import create_data_loader


def load_imagenet_data(train_dir, test_dir, batch_size, max_samples_per_class):
    """
    Loads the ImageNet dataset into train and test data loaders. There are two loader for the train dataset: one with
    augmentations and one without them, for testing purposes.
    :param train_dir: The name of the directory where the train dataset is located.
    :param test_dir: The name of the directory where the test dataset is located.
    :param batch_size: The batch size to use when loading data from both the train and test datasets.
    :param max_samples_per_class: Limit for the number of sample for each class in our datasets.
    :return: Data loaders for the train and test datasets of ImageNet (2 train, 1 test).
    """
    aug_transform, no_aug_transform = get_transforms()

    # Create a data loader for the train dataset with augmentations.
    train_dataset = ImageNetMini(root_dir=train_dir, transform=aug_transform, augment=True,
                                 max_samples_per_class=max_samples_per_class)
    train_loader = create_data_loader(train_dataset, is_train=True, batch_size=batch_size)

    # Create a data loader for the train dataset without augmentations.
    raw_train_dataset = ImageNetMini(root_dir=train_dir, transform=no_aug_transform, augment=False,
                                     max_samples_per_class=max_samples_per_class)
    raw_train_loader = create_data_loader(raw_train_dataset, is_train=True, batch_size=batch_size)

    # Create a data loader for the test dataset (without augmentations).
    test_dataset = ImageNetMini(root_dir=test_dir, transform=no_aug_transform, augment=False,
                                max_samples_per_class=max_samples_per_class)
    test_loader = create_data_loader(test_dataset, is_train=False, batch_size=batch_size)

    return train_loader, raw_train_loader, test_loader


class ImageNetMini(Dataset):
    """
    A small subset of the huge ImageNet dataset taken from https://www.kaggle.com/ifigotin/imagenetmini-1000.
    It has 34767 images as train set, and 3923 images as validation set (which we use as test set), divided almost
    evenly across 1000 different classes. We allow setting a limit for the number of samples for each class, in order
    to deal with the limitation of computational resources we have.
    """
    def __init__(self, root_dir, transform, augment, max_samples_per_class):
        self.augment = augment
        self.transform = transform
        self.class_mappings = {class_name: i for i, class_name in enumerate(os.listdir(root_dir))}
        images_path = np.array(sorted(os.path.join(root, file)
                                      for root, _, files in os.walk(root_dir) for file in files))
        labels = np.array([self.class_mappings[x.split(os.path.sep)[-2]] for x in images_path])

        # Reduce the number of samples per class.
        self.images_path = []
        self.labels = []
        for label in np.unique(labels):
            idx = np.where(labels == label)[0][:max_samples_per_class].astype(int)
            self.images_path += images_path[idx].tolist()
            self.labels += labels[idx].tolist()

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Load image from disk.
        img = Image.open(self.images_path[idx]).convert('RGB')
        # Get the image's label.
        label = self.labels[idx]

        if self.augment:
            # Training mode - create two augmentations.
            aug_image1 = self.transform(img)
            aug_image2 = self.transform(img)
            sample = {'image1': aug_image1, 'image2': aug_image2, 'label': label}
        else:
            # Test mode - no need for augmentations.
            image = self.transform(img)
            sample = {'image': image, 'label': label}
        return sample


def get_transforms():
    """
    Defines a transform to resize and normalize the test data, and defines a transform with some addition augmentations
    for the train data.
    :return: A transform with augmentations and a transform without augmentations.
    """
    # Create a transform with augmentations.
    aug_transform = transforms.Compose([
        transforms.Resize((250, 250)),  # resize image
        transforms.RandomHorizontalFlip(p=0.5),  # AUGMENTATION: Random Horizontal Flip
        transforms.RandomRotation(20),  # AUGMENTATION: Random Rotation
        transforms.RandomResizedCrop(224),  # AUGMENTATION: Random Cropping
        transforms.RandomGrayscale(p=0.1),  # AUGMENTATION: Random Greyscale
        transforms.ToTensor(),  # numpy array to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # normalize the image
        transforms.RandomErasing(p=1)  # AUGMENTATION: Random Erasing
    ])

    # Create a basic transform without augmentations.
    no_aug_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resize image
        transforms.ToTensor(),  # numpy array to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize the image
    ])

    return aug_transform, no_aug_transform
