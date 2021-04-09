import torch
from torch.utils.data import DataLoader


def create_data_loader(dataset, is_train, batch_size):
    """
    Create a data loader for the given dataset.
    :param dataset: The dataset to load.
    :param is_train: Whether the dataset will be used for training or testing.
    :param batch_size: The batch size to use.
    :return: Data loader for the given dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)


def create_sgd_optimizer(model, learning_rate, momentum):
    """
    Create a standard SGD optimizer for the given model.
    :param model: The model which the optimizer will use.
    :param learning_rate: The learning rate to use.
    :param momentum: The momentum to use.
    :return: SGD optimizer for the given model.
    """
    return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
