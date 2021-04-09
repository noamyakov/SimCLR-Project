import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


def load_cifar10_data():
    """
    Downloads the CIFAR10 dataset and loads it into train and test data loaders.
    :return: Data loaders for the train and test datasets of CIFAR10.
    """
    # Download the train and test datasets.
    train_data, test_data = download_dataset()

    # Create data loaders for the train and test datasets.
    train_loader = create_dataloader(train_data, is_train=True)
    test_loader = create_dataloader(test_data, is_train=False)
    return train_loader, test_loader


def train_model(model, optimizer, criterion, train_loader, test_loader, n_epochs, device):
    """
    Trains the model on the given train data and evaluate it on the given test data with every completed epoch.
    :param model: The model to train.
    :param optimizer: The optimizer used for train.
    :param criterion: The loss function used for training.
    :param train_loader: Data loader for the train dataset.
    :param test_loader: Data loader for the test dataset.
    :param n_epochs: The number of training epochs.
    :param device: The device to use for training.
    :return: The recorded losses and accuracies over the train and test datasets on every training epoch.
    """
    # Move model to device.
    model.to(device)

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    for epoch in range(n_epochs):
        # Train the model for one epoch and compute the average loss over the train set.
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
        # Compute the average loss over the test set after the current epoch.
        test_loss = compute_loss(model, criterion, test_loader, device)
        # Track train loss and test loss
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # Calculate the model's accuracy over the train and test sets after the current epoch.
        train_accuracy = evaluate_model(model, train_loader, device)
        test_accuracy = evaluate_model(model, test_loader, device)
        # Track train accuracy and test accuracy
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print_training_progress(epoch + 1, train_loss, test_loss, train_accuracy, test_accuracy)
    return train_losses, test_losses, train_accuracies, test_accuracies


def create_optimizer(model, learning_rate, momentum):
    """
    Create a standard SGD optimizer for the given model.
    :param model: The model which the optimizer will use.
    :param learning_rate: The learning rate to use.
    :param momentum: The momentum to use.
    :return: SGD optimizer for the given model.
    """
    return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


def create_criterion():
    """
    Create a Cross Entropy Loss object, which is great for multi-class classification.
    :return: Cross Entropy Loss object.
    """
    return nn.CrossEntropyLoss()


def replace_model_head(model, architecture, n_classes=10):
    """
    Replace the final layer of the given ResNet or VGG model with a fully connected layer.
    :param model: The model to replace its final layer.
    :param architecture: The model architecture, either 'resnet' or 'vgg'.
    :param n_classes: The number of output features in the new fully connected layer.
    """
    # Different model architectures use a different name for their final layer.
    if architecture == 'resnet':
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=in_features, out_features=n_classes)
    elif architecture == 'vgg':
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=in_features, out_features=n_classes)
    else:
        raise ValueError(f'Unsupported model architecture.')


def freeze_model(model):
    """
    Freeze all the given model's gradients.
    :param model: The model to freeze its gradients.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def evaluate_model(model, data_loader, device):
    """
    Evaluate a model by computing its top-1 accuracy over the given dataset.
    :param model: The model to compute its accuracy over the dataset.
    :param data_loader: Data loader for the dataset used for accuracy computation.
    :param device: The device to use for evaluation.
    :return: Top-1 accuracy of the model over the given dataset.
    """
    # Move model to device.
    model.to(device)
    # Set model in evaluation mode.
    model.eval()

    correct = 0
    for images, labels in data_loader:
        # Move data to device.
        images = images.to(device)
        labels = labels.to(device)

        # Predict using the model.
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        # Count the correct predictions.
        correct += (predicted == labels).sum().item()

    # Calculate accuracy.
    accuracy = correct / len(data_loader)
    return accuracy


def train_one_epoch(model, optimizer, criterion, data_loader, device):
    """
    Train a model on the given dataset for one epoch.
    :param model: The model to train.
    :param optimizer: The optimizer used for train.
    :param criterion: The loss function used for training.
    :param data_loader: Data loader for the train dataset.
    :param device: The device to use for training.
    :return: The average loss over the whole dataset.
    """
    # Move model to device.
    model.to(device)
    # Set model in training mode.
    model.train()

    total_loss = 0
    for images, labels in data_loader:
        # Move data to device.
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # Compute the loss.
        loss = criterion(outputs, labels)
        # Propagate gradients.
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    # Compute the average loss.
    average_loss = total_loss / len(data_loader)
    return average_loss


def compute_loss(model, criterion, data_loader, device):
    """
    Compute the average loss of the model over the given dataset.
    :param model: The model to compute its loss.
    :param criterion: The loss function used.
    :param data_loader: Data loader for the dataset.
    :param device: The device to use for computation.
    :return: The average loss over the whole dataset.
    """
    # Move model to device.
    model.to(device)
    # Set model in evaluation mode.
    model.eval()

    total_loss = 0
    for images, labels in data_loader:
        # Move data to device.
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # Compute the loss.
        loss = criterion(outputs, labels)
        total_loss += loss.item()
    return total_loss


def download_dataset(data_path='./CIFAR10_data'):
    """
    Download the CIFAR10 dataset.
    :param data_path: Where to save the downloaded data.
    :return: The train and test datasets of CIFAR10.
    """
    # Define a transform to resize and normalize the data.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Download the train and test datasets.
    train_data = CIFAR10(data_path, download=True, train=True, transform=transform)
    test_data = CIFAR10(data_path, download=True, train=False, transform=transform)
    return train_data, test_data


def create_dataloader(dataset, is_train, batch_size=64):
    """
    Create a data loader for the given dataset.
    :param dataset: The dataset to load.
    :param is_train: Whether the dataset will be used for training or testing.
    :param batch_size: The batch size to use.
    :return: Data loader for the given dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)


def print_training_progress(epoch, train_loss, test_loss, train_accuracy, test_accuracy, digits=3):
    """
    Prints the loss and accuracy results over the train and test dataset at the given training epoch.
    :param epoch: The training epoch to display its metrics.
    :param train_loss: The loss over the train dataset.
    :param test_loss: The loss over the test dataset.
    :param train_accuracy: The accuracy over the train dataset.
    :param test_accuracy: The accuracy over the test dataset.
    :param digits: The number of digits to display after the decimal point of every metric.
    """
    print('\t'.join([
        f'Epoch {epoch}',
        f'Train Loss: {train_loss:.{digits}f}', f'Test Loss: {test_loss:.{digits}f}',
        f'Train Accuracy: {train_accuracy:.{digits}f}', f'Test Accuracy: {test_accuracy:.{digits}f}'
    ]))
