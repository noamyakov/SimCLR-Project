import torch
from torch import nn

import utils
from training.self_supervised_learning import load_saved_simclr_models


def fine_tune_saved_simclr_models(architecture1, architecture2, train_loader, test_loader, n_classes, learning_rate,
                                  momentum, n_epochs):
    """
    Fine-tunes both of the two SimCLR models that were trained together and saved - one model at a time, and returns
    the recorded losses and accuracies over the train and test datasets on every training epoch, for each model
    separately. Then saves the two fine-tuned models (separately).
    :param architecture1: The first model architecture, one of 'ResNet18', 'ResNet34', 'VGG11' and 'VGG13'.
    :param architecture2: The second model architecture, one of 'ResNet18', 'ResNet34', 'VGG11' and 'VGG13'.
    :param train_loader: Data loader for the train dataset.
    :param test_loader: Data loader for the test dataset.
    :param n_classes: The number of output classes in the model's fine-tuned version.
    :param learning_rate: The learning rate to use for the optimizer.
    :param momentum: The momentum to use for the optimizer.
    :param n_epochs: The number of training epochs.
    :return: The recorded losses and accuracies over the train and test datasets on every training epoch, for each
    model separately.
    """
    # Load the saved SimCLR models.
    model1, model2 = load_saved_simclr_models(architecture1, architecture2,
                                              utils.construct_simclr_model_filename(architecture1, architecture2),
                                              utils.construct_simclr_model_filename(architecture2, architecture1))
    # Restore their base encoders (remove their projection head).
    utils.restore_base_encoder(model1, architecture1)
    utils.restore_base_encoder(model2, architecture2)

    # Fine-tune these two models - one at a time.
    model1_metrics = fine_tune(
        model1, architecture1, train_loader, test_loader, n_classes=n_classes, learning_rate=learning_rate,
        momentum=momentum, n_epochs=n_epochs
    )
    model2_metrics = fine_tune(
        model2, architecture2, train_loader, test_loader, n_classes=n_classes, learning_rate=learning_rate,
        momentum=momentum, n_epochs=n_epochs
    )

    # Save the fine-tuned models.
    utils.save_model(model1, utils.construct_fine_tuned_simclr_model_filename(architecture1, architecture2))
    utils.save_model(model2, utils.construct_fine_tuned_simclr_model_filename(architecture2, architecture1))

    # Return the losses and accuracies over the train and test datasets on every training epoch, for each model.
    return model1_metrics, model2_metrics


def fine_tune_pretrained_model(architecture, train_loader, test_loader, n_classes, learning_rate, momentum, n_epochs):
    """
    Fine-tunes a pre-trained model, and returns the recorded losses and accuracies over the train and test datasets on
    every training epoch. Then saves the fine-tuned model.
    :param architecture: The model architecture, one of 'ResNet18', 'ResNet34', 'VGG11' and 'VGG13'.
    :param train_loader: Data loader for the train dataset.
    :param test_loader: Data loader for the test dataset.
    :param n_classes: The number of output classes in the model's fine-tuned version.
    :param learning_rate: The learning rate to use for the optimizer.
    :param momentum: The momentum to use for the optimizer.
    :param n_epochs: The number of training epochs.
    :return: The recorded losses and accuracies over the train and test datasets on every training epoch, for each
    model separately.
    """
    # Load the pre-trained model.
    model = utils.load_model_based_on_architecture(architecture, pretrained=True)

    # Fine-tune this model.
    metrics = fine_tune(
        model, architecture, train_loader, test_loader, n_classes=n_classes, learning_rate=learning_rate,
        momentum=momentum, n_epochs=n_epochs
    )

    # Save the fine-tuned model.
    utils.save_model(model, utils.construct_fine_tuned_pretrained_model_filename(architecture))

    # Return the losses and accuracies over the train and test datasets on every training epoch.
    return metrics


def fine_tune(model, architecture, train_loader, test_loader, n_classes, learning_rate, momentum, n_epochs):
    """
    Fine-tunes the given model to deal with the given number of output classes, then trains it on the given train
    dataset, evaluates it (loss and accuracy) over both the train and the test datasets on every epoch, and returns
    these metrics.
    :param model: The model to fine-tune.
    :param n_epochs: The number of training epochs.
    :param architecture:
    :param train_loader: Data loader for the train dataset.
    :param test_loader: Data loader for the test dataset.
    :param n_classes: The number of output classes in the model's fine-tuned version.
    :param learning_rate: The learning rate to use for the optimizer.
    :param momentum: The momentum to use for the optimizer.
    :param n_epochs: The number of training epochs.
    :return: The recorded losses and accuracies over the train and test datasets on every training epoch.
    """
    # Freeze the gradients of all the model's parameters and so only the new head parameters can be modified.
    freeze_model(model)
    # Replace the models final layer (the head) with a new fully connected layer.
    replace_model_head(model, architecture, n_classes=n_classes)

    # Create a SGD optimizer for this model.
    optimizer = utils.create_sgd_optimizer([model], learning_rate=learning_rate, momentum=momentum)
    # Use the Cross Entropy Loss because is performs well with multi-class classification.
    criterion = create_cross_entropy_loss()
    # Pick the optimal device available for the training phase.
    device = utils.get_optimal_device()

    # Train the model on the given train dataset.
    metrics = train_model(
        model, optimizer, criterion, train_loader, test_loader, n_epochs=n_epochs, device=device
    )
    # Return the losses and accuracies over the train and test datasets on every training epoch.
    return metrics


def train_model(model, optimizer, criterion, train_loader, test_loader, n_epochs, device):
    """
    Trains the model on the given train data and evaluates it on the given test data with every completed epoch.
    :param model: The model to train.
    :param optimizer: The optimizer used for training.
    :param criterion: The loss function used for training.
    :param train_loader: Data loader for the train dataset.
    :param test_loader: Data loader for the test dataset.
    :param n_epochs: The number of training epochs.
    :param device: The device to use for training.
    :return: The recorded losses and accuracies over the train and test datasets on every training epoch.
    """
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    for epoch in range(n_epochs):
        # Train the model for one epoch and compute the average loss over the train set.
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
        # Compute the average loss over the test set after the current epoch.
        test_loss = compute_loss(model, criterion, test_loader, device)
        # Track train loss and test loss.
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # Calculate the model's accuracy over the train and test sets after the current epoch.
        train_accuracy = evaluate_model(model, train_loader, device)
        test_accuracy = evaluate_model(model, test_loader, device)
        # Track train accuracy and test accuracy.
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        utils.print_epoch_metrics(epoch + 1, {
            'Train Loss': train_loss,
            'Test Loss': test_loss,
            'Train Accuracy': train_accuracy,
            'Test Accuracy': test_accuracy
        })
    return train_losses, test_losses, train_accuracies, test_accuracies


def create_cross_entropy_loss():
    """
    Creates a Cross Entropy Loss object, which is great for multi-class classification.
    :return: Cross Entropy Loss object.
    """
    return nn.CrossEntropyLoss()


def replace_model_head(model, architecture, n_classes):
    """
    Replaces the final layer of the given ResNet or VGG model with a new fully connected layer.
    :param model: The model to replace its final layer.
    :param architecture: The model architecture, one of 'ResNet18', 'ResNet34', 'VGG11' and 'VGG13'.
    :param n_classes: The number of output features in the new fully connected layer.
    """
    # Load a model of the same architecture that will act as a skeleton and let us know how many in_features there
    # are in the final layer (the head).
    base_encoder_skeleton = utils.load_model_based_on_architecture(architecture, pretrained=False)
    final_layer_skeleton = utils.get_final_layer_based_on_architecture(base_encoder_skeleton, architecture)
    in_features = final_layer_skeleton.in_features

    # Create a new fully connected layer and set it as the models' head.
    fc = nn.Linear(in_features=in_features, out_features=n_classes)
    utils.set_final_layer_based_on_architecture(model, fc, architecture)


def freeze_model(model):
    """
    Freezes all the given model's gradients.
    :param model: The model to freeze its gradients.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def evaluate_model(model, data_loader, device):
    """
    Evaluates a model by computing its top-1 accuracy over the given dataset.
    :param model: The model to compute its accuracy over the dataset.
    :param data_loader: Data loader for the dataset used for accuracy computation.
    :param device: The device to use for evaluation.
    :return: Top-1 accuracy of the model over the given dataset.
    """
    # Move model to device.
    model.to(device)
    # Set model in evaluation mode.
    model.eval()

    total = 0
    correct = 0
    for images, labels in data_loader:
        # Move data to device.
        images = images.to(device)
        labels = labels.to(device)

        # Predict using the model.
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += len(labels)
        # Count the correct predictions.
        correct += (predicted == labels).sum().item()

    # Calculate accuracy.
    accuracy = correct / total
    return accuracy


def train_one_epoch(model, optimizer, criterion, data_loader, device):
    """
    Trains a model on the given dataset for one epoch.
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
    Computes the average loss of the model over the given dataset.
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
    # Compute the average loss.
    average_loss = total_loss / len(data_loader)
    return average_loss
