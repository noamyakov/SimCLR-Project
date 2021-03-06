import itertools

import torch
from torch.utils.data import DataLoader
from torchvision import models


def load_model_based_on_architecture(architecture, pretrained):
    """
    Loads a PyTorch pretrained model with the given architecture.
    :param architecture: The model architecture, one of 'ResNet18', 'ResNet34', 'VGG11' and 'VGG13'.
    :param pretrained: Whether the model should be pre-trained or not.
    :return: A PyTorch pretrained model that matches the given architecture.
    """
    if architecture == 'ResNet18':
        return models.resnet18(pretrained=pretrained)
    elif architecture == 'ResNet34':
        return models.resnet34(pretrained=pretrained)
    elif architecture == 'VGG11':
        return models.vgg11(pretrained=pretrained)
    elif architecture == 'VGG13':
        return models.vgg13(pretrained=pretrained)
    else:
        raise ValueError(f'Unsupported model architecture: {architecture}')


def create_sgd_optimizer(models_in_use, learning_rate, momentum):
    """
    Creates a standard SGD optimizer for all of the given models.
    :param models_in_use: The models which the optimizer will use their parameters.
    :param learning_rate: The learning rate to use.
    :param momentum: The momentum to use.
    :return: SGD optimizer for all of the given models.
    """
    # Chain the parameters of all the models.
    parameters = itertools.chain(*[model.parameters() for model in models_in_use])
    # Create a SGD optimizer for this parameters.
    return torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum)


def get_optimal_device():
    """
    Returns the optimal device available, either CUDA or CPU.
    :return: The optimal device available.
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def save_model(model, path):
    """
    Saves a model at the given path.
    :param model: The model to save.
    :param path: Where to save the model.
    """
    torch.save(model.state_dict(), path)


def restore_base_encoder(model, architecture):
    """
    Restores the base encoder (original form) of the given SimCLR model. Removes the projection head.
    :param model: The model to restore its base encoder.
    :param architecture: The model architecture, one of 'ResNet18', 'ResNet34', 'VGG11' and 'VGG13'.
    """
    # Load a model of the same architecture that will act as a skeleton and "donate" its final layer for the model.
    base_encoder_skeleton = load_model_based_on_architecture(architecture, pretrained=False)
    final_layer_skeleton = get_final_layer_based_on_architecture(base_encoder_skeleton, architecture)

    # Set the skeleton's final layer as the final layer of the model.
    set_final_layer_based_on_architecture(model, final_layer_skeleton, architecture)


def create_data_loader(dataset, is_train, batch_size):
    """
    Creates a data loader for the given dataset.
    :param dataset: The dataset to load.
    :param is_train: Whether the dataset will be used for training or testing.
    :param batch_size: The batch size to use.
    :return: Data loader for the given dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)


def get_final_layer_based_on_architecture(model, architecture):
    """
    Returns the final layer (the head) of the given model based on its architecture.
    :param model: The model to return its final layer.
    :param architecture: The model architecture, one of 'ResNet18', 'ResNet34', 'VGG11' and 'VGG13'.
    :return: The model's final layer.
    """
    # Different model architectures use a different name for their final layer.
    if architecture.startswith('ResNet'):
        return model.fc
    elif architecture.startswith('VGG'):
        return model.classifier[6]
    else:
        raise ValueError(f'Unsupported model architecture: {architecture}')


def set_final_layer_based_on_architecture(model, layer, architecture):
    """
    Sets the given layer as the final layer (the head) of the given model based on its architecture.
    :param model: The model to sets its final layer.
    :param layer: The layer to set as the model's final layer.
    :param architecture: The model architecture, one of 'ResNet18', 'ResNet34', 'VGG11' and 'VGG13'.
    """
    # Different model architectures use a different name for their final layer.
    if architecture.startswith('ResNet'):
        model.fc = layer
    elif architecture.startswith('VGG'):
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
    print('\t'.join([f'Epoch {epoch}:', *[f'{metric}: {value:.{digits}f}' for metric, value in metrics.items()]]))


def construct_simclr_model_filename(architecture_of_this, architecture_of_other):
    """
    Constructs a filename in which our SimCLR model will be saved.
    :param architecture_of_this: The model architecture of the SimCLR model the returned filename is used for.
    :param architecture_of_other: The model architecture of the other SimCLR model that was trained together with the
    model the returned filename is used for.
    :return: The filename in which our SimCLR model will be saved.
    """
    return f'{architecture_of_this}_trained_with_{architecture_of_other}.pth'


def construct_fine_tuned_simclr_model_filename(architecture_of_this, architecture_of_other):
    """
    Constructs a filename in which our fine-tuned SimCLR model will be saved.
    :param architecture_of_this: The model architecture of the fine-tuned SimCLR model the returned filename is used
    for.
    :param architecture_of_other: The model architecture of the other SimCLR model that was trained together with the
    model the returned filename is used for.
    :return: The filename in which our fine-tuned SimCLR model will be saved.
    """
    return f'fine-tuned_{construct_simclr_model_filename(architecture_of_this, architecture_of_other)}'


def construct_fine_tuned_pretrained_model_filename(architecture):
    """
    Constructs a filename in which our fine-tuned pre-trained model will be saved.
    :param architecture: The model architecture of the fine-tuned pre-trained model the returned filename is used for.
    :return: The filename in which our fine-tuned pre-trained model will be saved.
    """
    return f'fine-tuned_pre-trained_{architecture}.pth'
