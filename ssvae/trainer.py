import numpy as np
import torch


def run_inference_for_epoch(data_loaders, losses, cuda=False):
    """Run one epoch of SVI over interleaved supervised and unsupervised batches.

    Supervised batches pass labels to the loss; unsupervised batches withhold them
    so the guide must infer the phase label from engineered features.
    """
    num_losses = len(losses)
    sup_batches = len(data_loaders["sup"])
    unsup_batches = len(data_loaders["unsup"])

    epoch_losses_sup = [0.0] * num_losses
    epoch_losses_unsup = [0.0] * num_losses

    sup_iter = iter(data_loaders["sup"])
    unsup_iter = iter(data_loaders["unsup"])

    schedule = np.random.permutation([True] * sup_batches + [False] * unsup_batches)

    for is_supervised in schedule:
        if is_supervised:
            xs, es, ys = next(sup_iter)
        else:
            xs, es, ys = next(unsup_iter)

        if cuda:
            xs, es, ys = xs.cuda(), es.cuda(), ys.cuda()

        batchsize = xs.size(0)
        xs = xs.view(batchsize, -1)
        es = es.view(batchsize, -1)

        for loss_id, loss in enumerate(losses):
            if is_supervised:
                epoch_losses_sup[loss_id] += loss.step(xs, es, ys)
            else:
                epoch_losses_unsup[loss_id] += loss.step(xs, es)

    return epoch_losses_sup, epoch_losses_unsup


def evaluate_model(data_loader, model, losses, cuda=False):
    """Compute ELBO losses on a labelled data loader without parameter updates."""
    model.eval()
    epoch_losses = [0.0] * len(losses)

    for xs, es, ys in data_loader:
        if cuda:
            xs, es, ys = xs.cuda(), es.cuda(), ys.cuda()
        batchsize = xs.size(0)
        xs = xs.view(batchsize, -1)
        es = es.view(batchsize, -1)
        for loss_id, loss in enumerate(losses):
            epoch_losses[loss_id] += loss.evaluate_loss(xs, es, ys)

    return epoch_losses


def get_accuracy(data_loader, classifier_fn, cuda=False):
    """Compute classification accuracy over a DataLoader.

    Returns:
        accuracy (float), actuals (Tensor), predictions (Tensor)
    """
    predictions, actuals = [], []
    for xs, es, ys in data_loader:
        if cuda:
            xs, es, ys = xs.cuda(), es.cuda(), ys.cuda()
        predictions.append(classifier_fn(es))
        actuals.append(ys)
    actuals = torch.cat(actuals, dim=0).squeeze()
    predictions = torch.cat(predictions, dim=0).squeeze()
    accuracy = (actuals == predictions).float().mean()
    return accuracy.item(), actuals, predictions
