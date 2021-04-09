import torch
from torch import nn
from torchvision import models

import utils


def create_simclr_models(architecture1, architecture2):
    # Create the base encoders.
    model1 = load_pretrained_model_based_on_architecture(architecture1)
    model2 = load_pretrained_model_based_on_architecture(architecture2)

    # Change the models' classifiers to a shared projection head.
    in_features1 = utils.get_final_layer_based_on_architecture(model1, architecture1).in_features
    in_features2 = utils.get_final_layer_based_on_architecture(model2, architecture2).in_features
    if in_features1 == in_features2:
        head1 = nn.Sequential(nn.Linear(in_features1, 100),
                              nn.ReLU(),
                              nn.Linear(100, 40),
                              nn.ReLU(),
                              nn.Linear(40, 20))
        head2 = head1
    else:
        # In case of mismatch, the two models won't share their projection head's first layer.
        shared_head = nn.Sequential(nn.ReLU(),
                                    nn.Linear(100, 40),
                                    nn.ReLU(),
                                    nn.Linear(40, 20))
        head1 = nn.Sequential(nn.Linear(in_features1, 100), shared_head)
        head2 = nn.Sequential(nn.Linear(in_features2, 100), shared_head)

    utils.set_final_layer_based_on_architecture(model1, head1, architecture1)
    utils.set_final_layer_based_on_architecture(model2, head2, architecture2)
    return model1, model2


def self_supervised_training(model1, model2, optimizer, train_loader, temperature, n_epochs, device):
    train_losses = []
    for epoch in range(n_epochs):
        # Train the model for one epoch and compute the average loss over the train set.
        train_loss = train_one_epoch(model1, model2, optimizer, train_loader, temperature, device)
        # Track train loss.
        train_losses.append(train_loss)

        utils.print_epoch_metrics(epoch + 1, {
            'Contrastive Loss': train_loss
        })
    return train_losses


def train_one_epoch(model1, model2, optimizer, data_loader, temperature, device):
    """
    Trains two models (simultaneously) on the given dataset for one epoch.
    :param model1: The first model to train.
    :param model2: The second model to train.
    :param optimizer: The optimizer used for train.
    :param data_loader: Data loader for the train dataset.
    :param temperature:
    :param device: The device to use for training.
    :return: The average loss over the whole dataset.
    """
    # Move models to device.
    model1.to(device)
    model2.to(device)
    # Set models in training mode.
    model1.train()
    model2.train()

    total_loss = 0
    for aug_batch in data_loader:
        # Move data to device.
        images1 = aug_batch['image1'].to(device)
        images2 = aug_batch['image2'].to(device)

        optimizer.zero_grad()
        outputs1 = model1(images1)
        outputs2 = model2(images2)

        # Compute the loss.
        loss = contrastive_loss(outputs1, outputs2, temperature)
        # Propagate gradients.
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    # Compute the average loss.
    average_loss = total_loss / len(data_loader)
    return average_loss


def contrastive_loss(aug1_batch, aug2_batch, temperature):
    """
    Calculates the contrastive loss between the two given batches.
    :param aug1_batch: The first batch.
    :param aug2_batch: The second batch.
    :param temperature: Hyper-parameter that scales the contrastive loss.
    :return: The contrastive loss between the two given batches.
    """
    batch_size = len(aug1_batch)

    # Concatenate both batches to one big batch.
    batch = torch.cat((aug1_batch.unsqueeze(1), aug2_batch.unsqueeze(1)), dim=1).reshape(2 * batch_size, -1)

    # Compute similarity between each pair in representations matrix.
    sim = pairwise_cosine_sim(batch, batch)
    # Divide values by temperature.
    sim /= temperature

    # Calculate the denominator in the loss: Apply exponent on all values.
    exp_sim = torch.exp(sim)
    # Zero out similarities between samples to themselves (remove the diagonal) and calculate the sum for each
    # row in the matrix.
    sum_exp_sim_rows = torch.sum(exp_sim * (1 - torch.eye(2 * batch_size, device=exp_sim.device)), dim=0)

    # Calculate the log probabilities using the log rule: log(e^x/sum(y)) = x - log(sum(y)).
    log_prob = sim - torch.log(sum_exp_sim_rows).reshape(-1, 1)
    # Take only the positive cells where the samples should be similar. We want to maximize the similarity between
    # them.
    log_prob = torch.cat((log_prob.diag(1)[::2], log_prob.diag(-1)[::2]))

    # Multiply by -1 and compute the mean (sum of log probabilities/(2*batch_size)).
    loss = -log_prob.mean()
    return loss


def pairwise_cosine_sim(X, Y):
    """
    Computes the cosine similarity between each pair in the two given batches of vectors.
    :param X: The first batch of vectors.
    :param Y: The second batch of vectors.
    :return: A matrix with shape [|X|,|Y|] where cell [i,j] is the cosine similarity between X[i,:] and Y[j,:].
    """
    X = X / torch.linalg.norm(X, dim=1).reshape(-1, 1)
    Y = Y / torch.linalg.norm(Y, dim=1).reshape(-1, 1)
    return torch.mm(X, Y.T)


def load_pretrained_model_based_on_architecture(architecture):
    """
    Loads a PyTorch pretrained model with the given architecture (either ResNet18, ResNet34, VGG11 or VGG13).
    :param architecture: The model architecture, one of 'resnet18', 'resnet34', 'vgg11' and 'vgg13'.
    :return: A PyTorch pretrained model that matches the given architecture.
    """
    if architecture == 'resnet18':
        return models.resnet18(pretrained=True)
    elif architecture == 'resnet34':
        return models.resnet34(pretrained=True)
    elif architecture == 'vgg11':
        return models.vgg11(pretrained=True)
    elif architecture == 'vgg13':
        return models.vgg13(pretrained=True)
    else:
        raise ValueError(f'Unsupported model architecture.')
