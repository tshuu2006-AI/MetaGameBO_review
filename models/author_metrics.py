"""Deterministic audit of the public classifier, separate from its objective."""
import torch
from metrics import summarize_weighted_bce


@torch.no_grad()
def evaluate_classifier(model, dataset, loss_fn, alpha=0.8):
    was_training = model.training
    model.eval()
    try:
        # DataLoader draws a base seed even without shuffling. Keep metric audit
        # from consuming the training RNG used by upstream embedding noise.
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset),
                                            generator=torch.Generator().manual_seed(0))
        X, ids, labels, weights = next(iter(loader))
        output = model(X, ids)
        result = summarize_weighted_bce(output[1], labels, weights, ids, alpha)
        regularization = float(.1 * loss_fn._embedding_loss(model.task_embedding.weight[1:]))
        result['regularization'] = regularization
        result['objective_with_regularization'] = result['micro_bce'] + regularization
        return result
    finally:
        model.train(was_training)
