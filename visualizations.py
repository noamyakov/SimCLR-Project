import matplotlib.pyplot as plt


def plot_epochs_metrics(metrics, unit_of_measurement):
    """
    Plots all the given metrics about epochs in a single grid, where every metric is shown with a graph.
    :param metrics: Mapping between metric names and their corresponding list of values - one value per epoch.
    :param unit_of_measurement: The unit in which we measure all the metrics - this is used for naming the Y axis.
    """
    # Computes the total number of epochs: the length of the longest metric list of values.
    n_epochs = len(max(metrics.values(), key=len))
    total_epochs = range(1, n_epochs + 1)

    # Plot each metric in its own graph, all on the same grid.
    for metric, values in metrics.items():
        # A metric does not have to be computed for all the epochs, but it should start with the first epoch and cannot
        # skip over any epoch.
        epochs_for_metric = range(1, len(values) + 1)
        plt.plot(epochs_for_metric, values, 'o-', label=metric)

    plt.xticks(total_epochs)
    plt.xlabel('Epochs')
    plt.ylabel(unit_of_measurement)
    plt.title(f'{unit_of_measurement} Along Epochs')
    plt.legend()
    plt.show()
