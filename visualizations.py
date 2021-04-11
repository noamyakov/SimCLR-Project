import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

import utils
from training.self_supervised_learning import extract_features_and_labels


def plot_metrics_along_epochs(metrics, unit_of_measurement):
    """
    Plots all the given metrics about epochs in a single grid, where every metric is shown with a graph.
    :param metrics: Mapping between metric names and their corresponding list of values - one value per epoch.
    :param unit_of_measurement: The unit in which we measure all the metrics - this is used for naming the Y axis.
    """
    # Computes the total number of epochs: the length of the longest metric list of values.
    n_epochs = len(max(metrics.values(), key=len))
    total_epochs = range(1, n_epochs + 1)

    plt.rcParams['figure.figsize'] = (13, 5)
    # Plot each metric in its own graph, all on the same grid.
    for metric, values in metrics.items():
        # A metric does not have to be computed for all the epochs, but it should start with the first epoch and cannot
        # skip over any epoch.
        metric_epochs = range(1, len(values) + 1)
        plt.plot(metric_epochs, values, 'o-', label=metric)

    plt.xticks(total_epochs)
    plt.xlabel('Epoch')
    plt.ylabel(unit_of_measurement)
    plt.title(f'{unit_of_measurement} Along Epochs')
    plt.legend()
    plt.show()


def simclr_features_embedding(architecture1, architecture2, train_loader, test_loader, max_n_classes):
    """
    Extract the features of the given train and test datasets (without augmentations) that are produced by 2 trained
    SimCLR models, and plot their embeddings in a 2-dimensional space.
    :param architecture1: The first model architecture, one of 'ResNet18', 'ResNet34', 'VGG11' and 'VGG13'.
    :param architecture2: The second model architecture, one of 'ResNet18', 'ResNet34', 'VGG11' and 'VGG13'.
    :param train_loader: Data loader for the train dataset.
    :param test_loader: Data loader for the test dataset.
    :param max_n_classes: The maximal number of classes to embed their features.
    """
    # Load the saved SimCLR models.
    model1 = utils.load_saved_model(architecture1, utils.construct_simclr_model_filename(architecture1, architecture2))
    model2 = utils.load_saved_model(architecture2, utils.construct_simclr_model_filename(architecture2, architecture1))

    # Pick the optimal device available for the feature extraction phase.
    device = utils.get_optimal_device()
    # Plot the embedded features produced by each of the models.
    plot_embedded_features(model1, architecture1, train_loader, test_loader, max_n_classes=max_n_classes, device=device)
    plot_embedded_features(model2, architecture2, train_loader, test_loader, max_n_classes=max_n_classes, device=device)


def plot_embedded_features(model, architecture, train_loader, test_loader, max_n_classes, device):
    """
    Extract the features of the given train and test datasets (without augmentations) that are produced by the given
    trained SimCLR model, and plot their embeddings in a 2-dimensional space.
    :param model: The model to use for features extraction.
    :param architecture: The model architecture, one of 'ResNet18', 'ResNet34', 'VGG11' and 'VGG13'.
    :param train_loader: Data loader for the train dataset.
    :param test_loader: Data loader for the test dataset.
    :param max_n_classes: The maximal number of classes to embed their features.
    :param device: The device to use for features extraction.
    """
    # Extract train and test features that were produced by the first model.
    train_features, train_labels = extract_features_and_labels(model, train_loader, device=device)
    test_features, test_labels = extract_features_and_labels(model, test_loader, device=device)

    # Reduce features dimensions with T-SNE (from 20d vectors to 2d vectors)
    tsne = TSNE(n_components=2, random_state=0)
    train_embedded = tsne.fit_transform(train_features)
    test_embedded = tsne.fit_transform(test_features)

    # Get the number of classes to embed their features.
    classes = np.unique(train_labels)[:max_n_classes]

    plt.rcParams['figure.figsize'] = (13, 5)
    # Plot the embedded features of the train dataset.
    plt.subplot(1, 2, 1)
    for c in classes:
        idx = np.where(train_labels == c)
        plt.scatter(train_embedded[idx, 0], train_embedded[idx, 1], label=c)
        plt.title(f'Train Features Produced by {architecture}')
    # Plot the embedded features of the test dataset.
    plt.subplot(1, 2, 2)
    for c in classes:
        idx = np.where(test_labels == c)
        plt.scatter(test_embedded[idx, 0], test_embedded[idx, 1], label=c)
        plt.title(f'Test Features Produced by {architecture}')
    plt.show()
