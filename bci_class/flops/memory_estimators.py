"""Estimators of amount of memory taken up by the models."""


def compute_parameter_count_original_model(
    *,
    # number of channels in a single timestep (e.g. 512)
    neural_dim: int,
    # number of hidden units in each recurrent layer - equal to the size of the hidden state
    n_units: int,
    # the number of timesteps to concat on initial input layer
    patch_size: int,
    # number of recurrent layers 
    n_layers: int,
    # number of classes 
    n_classes: int,
) -> int:
    per_day_params = neural_dim**2 + neural_dim

    gru_params = 0
    for layer_index in range(n_layers):
        dx = neural_dim * patch_size if layer_index == 0 else n_units
        gru_params += 3 * n_units * dx + 3 * n_units
        gru_params += 3 * n_units**2 + 3 * n_units

    out_params = n_units * n_classes + n_classes

    return per_day_params + gru_params + out_params


def compute_parameter_count_rank_reduced(
    *,
    # number of channels in a single timestep (e.g. 512)
    neural_dim: int,
    # number of hidden units in each recurrent layer - equal to the size of the hidden state
    n_units: int,
    # the number of timesteps to concat on initial input layer
    patch_size: int,
    # number of recurrent layers 
    n_layers: int,
    # number of classes 
    n_classes: int,

    # Rank-reduction flags.
    day_weights_rank: int,
    gru_rank: int,
    gru_rank_ih_l0: int,
) -> int:
    per_day_params = 2 * neural_dim * day_weights_rank + neural_dim

    gru_params = 0
    for layer_index in range(n_layers):
        dx = neural_dim * patch_size if layer_index == 0 else n_units
        rkx = gru_rank_ih_l0 if layer_index == 0 else gru_rank
        gru_params += 3 * n_units * rkx + 2 * rkx * dx + 3 * n_units
        gru_params += 3 * n_units * gru_rank + 2 * gru_rank * n_units + 3 * n_units

    out_params = n_units * n_classes + n_classes

    return per_day_params + gru_params + out_params
