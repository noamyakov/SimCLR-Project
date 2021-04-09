from torchvision import transforms
from torchvision.datasets import CIFAR10

from utils import create_data_loader


def load_cifar10_data(batch_size):
    """
    Downloads the CIFAR10 dataset and loads it into train and test data loaders.
    :param batch_size: The batch size to use when loading data from the dataset.
    :return: Data loaders for the train and test datasets of CIFAR10.
    """
    # Download the train and test datasets.
    train_data, test_data = download_dataset()

    # Create data loaders for the train and test datasets.
    train_loader = create_data_loader(train_data, is_train=True, batch_size=batch_size)
    test_loader = create_data_loader(test_data, is_train=False, batch_size=batch_size)
    return train_loader, test_loader


def download_dataset(data_path='./CIFAR10_data'):
    """
    Downloads the CIFAR10 dataset.
    :param data_path: Where to save the downloaded data.
    :return: The train and test datasets of CIFAR10.
    """
    transform = get_transform()

    # Download the train and test datasets.
    train_data = CIFAR10(data_path, download=True, train=True, transform=transform)
    test_data = CIFAR10(data_path, download=True, train=False, transform=transform)
    return train_data, test_data


def get_transform():
    """
    Defines a transform to resize and normalize the data.
    :return: The created transform.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
