"""Unstructured pruning based on magnitudes of weights."""
import collections
from typing import List

import torch

from model_training import rnn_model


@torch.no_grad()
def prune_day_weights_by_magnitude(model: 'rnn_model.GRUDecoder', retain_fraction: float):
    """Sets day_weights with the lowest (1-retain_fraction) fraction of magnitudes to zero.

    This will do it for each day's weights independently.
    """
    index_to_params = collections.defaultdict(list)

    for n, p in model.named_parameters():
        if not _is_day_weight(n):
            continue
        index = int(n.split('.')[-1])
        index_to_params[index].append(p)

    for parameters in index_to_params.values():
        _prune_parameters(parameters, retain_fraction)


@torch.no_grad()
def prune_non_day_weights_by_magnitude(model: 'rnn_model.GRUDecoder', retain_fraction: float):
    """Sets weights other than the day_weights with the lowest (1-retain_fraction) fraction of magnitudes to zero."""
    parameters = [p for n, p in model.named_parameters() if not _is_day_weight(n)]
    _prune_parameters(parameters, retain_fraction)


def _is_day_weight(n: str) -> bool:
    return n.startswith('day_weights.') or n.startswith('day_biases.')


def _prune_parameters(parameters: List[torch.nn.Parameter], retain_fraction: float):
    """Prunes the parameters in place."""
    v = torch.nn.utils.parameters_to_vector(parameters)

    keep_k = int(retain_fraction * v.numel())
    _, keep_inds = torch.topk(v.abs(), k=keep_k)

    keep_vals = v[keep_inds]

    v.zero_()
    v[keep_inds] = keep_vals

    torch.nn.utils.vector_to_parameters(v, parameters)
