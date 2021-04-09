import torch
from torch.utils.data import DataLoader


def create_data_loader(dataset, is_train, batch_size):
    """
    Creates a data loader for the given dataset.
    :param dataset: The dataset to load.
    :param is_train: Whether the dataset will be used for training or testing.
    :param batch_size: The batch size to use.
    :return: Data loader for the given dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)


def create_sgd_optimizer(model, learning_rate, momentum):
    """
    Creates a standard SGD optimizer for the given model.
    :param model: The model which the optimizer will use.
    :param learning_rate: The learning rate to use.
    :param momentum: The momentum to use.
    :return: SGD optimizer for the given model.
    """
    return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


def get_final_layer_based_on_architecture(model, architecture):
    """
    Returns the final layer (the head) of the given model based on its architecture.
    :param model: The model to return its final layer.
    :param architecture: The model architecture, either 'resnetX' or 'vggX'.
    :return: The model's final layer.
    """
    # Different model architectures use a different name for their final layer.
    if architecture.startswith('resnet'):
        return model.fc
    elif architecture.startswith('vgg'):
        return model.classifier[6]
    else:
        raise ValueError(f'Unsupported model architecture: {architecture}')


def set_final_layer_based_on_architecture(model, layer, architecture):
    """
    Sets the given layer as the final layer (the head) of the given model based on its architecture.
    :param model: The model to sets its final layer.
    :param layer: The layer to set as the model's final layer.
    :param architecture: The model architecture, either 'resnetX' or 'vggX'.
    """
    # Different model architectures use a different name for their final layer.
    if architecture.startswith('resnet'):
        model.fc = layer
    elif architecture.startswith('vgg'):
        model.classifier[6] = layer
    else:
        raise ValueError(f'Unsupported model architecture: {architecture}')


def print_epoch_metrics(epoch, metrics, digits=3):
    """
    Prints metrics of the given training epoch.
    :param epoch: The training epoch to display its metrics.
    :param metrics: The metrics to display about the epoch.
    :param digits: The number of digits to display after the decimal point of every metric.
    """
    print('\t'.join([f'Epoch {epoch}', *[f'{metric}: {value:.{digits}f}' for metric, value in metrics.items()]]))
