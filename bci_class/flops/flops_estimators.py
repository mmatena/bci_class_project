"""Estimators of flops per patch.

Counts each element-wise non-linearity as a single FLOP per element.

"""
import dataclasses
from typing import Tuple

###############################################################################


@dataclasses.dataclass
class FlopsInfo:

    # the number of timesteps to stride over when concatenating initial input 
    patch_stride: int

    # The number of FLOPs that per-day preprocessing a single timestamp's worth of
    # data takes.
    per_sample_day_flops: int

    # Number of FLOPs taken by processing for each layer in the GRU for a single patch..
    per_patch_gru_flops_by_layer: Tuple[int, ...]

    # The number of FLOPs taken to process a patch by the output network.
    per_patch_output_flops: int

    #######################################################

    @property
    def per_patch_gru_flops(self) -> int:
        return sum(self.per_patch_gru_flops_by_layer)

    @property
    def per_patch_steady_state_flops(self) -> int:
        """Total FLOPS needed to process a patch.

        Assumes that we are not the first patch and can reuse the per-day preprocessing
        done by the previous patch.
        """
        return self.patch_stride * self.per_sample_day_flops + self.per_patch_output_flops + self.per_patch_gru_flops


###############################################################################
# Stuff for the original model, i.e. with no compression.


def compute_flops_original_model(
    *,
    # number of channels in a single timestep (e.g. 512)
    neural_dim: int,
    # number of hidden units in each recurrent layer - equal to the size of the hidden state
    n_units: int,
    # the number of timesteps to concat on initial input layer
    patch_size: int,
    # the number of timesteps to stride over when concatenating initial input 
    patch_stride: int,
    # number of recurrent layers 
    n_layers: int,
    # number of classes 
    n_classes: int,
) -> FlopsInfo:
    # The per-day processing of the neural data.

    # matmul + bias + non-linearity
    per_sample_day_flops = 2 * neural_dim**2 + neural_dim + neural_dim

    # The GRU.

    per_patch_gru_flops_by_layer = []
    for layer_index in range(n_layers):
        dx = neural_dim * patch_size if layer_index == 0 else n_units

        x_affine_flops = 2 * 3 * n_units * dx + 3 * n_units
        h_affine_flops = 2 * 3 * n_units**2 + 3 * n_units

        # sum + non-linearity
        r_extra_flops = 2 * n_units
        # sum + non-linearity
        z_extra_flops = 2 * n_units
        # hadamard product + sum + non-linearity
        n_extra_flops = 3 * n_units
        h_extra_flops = 4 * n_units

        per_patch_gru_flops_by_layer.append(sum([
            x_affine_flops,
            h_affine_flops,
            r_extra_flops,
            z_extra_flops,
            n_extra_flops,
            h_extra_flops,
        ]))

    # Output layer

    per_patch_output_flops = 2 * n_units * n_classes + n_classes
    
    return FlopsInfo(
        patch_stride=patch_stride,
        per_sample_day_flops=per_sample_day_flops,
        per_patch_gru_flops_by_layer=tuple(per_patch_gru_flops_by_layer),
        per_patch_output_flops=per_patch_output_flops,
    )

###############################################################################
# Stuff for the rank-reduced model.


def compute_flops_rank_reduced(
    *,
    # number of channels in a single timestep (e.g. 512)
    neural_dim: int,
    # number of hidden units in each recurrent layer - equal to the size of the hidden state
    n_units: int,
    # the number of timesteps to concat on initial input layer
    patch_size: int,
    # the number of timesteps to stride over when concatenating initial input 
    patch_stride: int,
    # number of recurrent layers 
    n_layers: int,
    # number of classes 
    n_classes: int,

    # Rank-reduction flags.
    day_weights_rank: int,
    gru_rank: int,
    gru_rank_ih_l0: int,
) -> FlopsInfo:
    # The per-day processing of the neural data.

    # matmul + bias + non-linearity
    per_sample_day_flops = 4 * neural_dim * day_weights_rank + neural_dim + neural_dim

    # The GRU.

    per_patch_gru_flops_by_layer = []
    for layer_index in range(n_layers):
        dx = neural_dim * patch_size if layer_index == 0 else n_units
        rkx = gru_rank_ih_l0 if layer_index == 0 else gru_rank

        x_affine_flops = 2 * 3 * n_units * rkx + 2 * rkx * dx + 3 * n_units
        h_affine_flops = 2 * 3 * n_units * gru_rank + 2 * gru_rank * n_units + 3 * n_units

        # sum + non-linearity
        r_extra_flops = 2 * n_units
        # sum + non-linearity
        z_extra_flops = 2 * n_units
        # hadamard product + sum + non-linearity
        n_extra_flops = 3 * n_units
        h_extra_flops = 4 * n_units

        per_patch_gru_flops_by_layer.append(sum([
            x_affine_flops,
            h_affine_flops,
            r_extra_flops,
            z_extra_flops,
            n_extra_flops,
            h_extra_flops,
        ]))

    # Output layer

    per_patch_output_flops = 2 * n_units * n_classes + n_classes
    
    return FlopsInfo(
        patch_stride=patch_stride,
        per_sample_day_flops=per_sample_day_flops,
        per_patch_gru_flops_by_layer=tuple(per_patch_gru_flops_by_layer),
        per_patch_output_flops=per_patch_output_flops,
    )


###############################################################################
# Stuff for the pruned model. This is just an estimate.


def compute_flops_rank_pruned(
    *,
    # number of channels in a single timestep (e.g. 512)
    neural_dim: int,
    # number of hidden units in each recurrent layer - equal to the size of the hidden state
    n_units: int,
    # the number of timesteps to concat on initial input layer
    patch_size: int,
    # the number of timesteps to stride over when concatenating initial input 
    patch_stride: int,
    # number of recurrent layers 
    n_layers: int,
    # number of classes 
    n_classes: int,

    # Rank-reduction flags.
    retain_fraction: float,
) -> FlopsInfo:
    # The per-day processing of the neural data.

    # matmul + bias + non-linearity
    per_sample_day_flops = 2 * retain_fraction * neural_dim**2 + retain_fraction * neural_dim + neural_dim

    # The GRU.

    per_patch_gru_flops_by_layer = []
    for layer_index in range(n_layers):
        dx = neural_dim * patch_size if layer_index == 0 else n_units

        x_affine_flops = retain_fraction * (2 * 3 * n_units * dx + 3 * n_units)
        h_affine_flops = retain_fraction * (2 * 3 * n_units**2 + 3 * n_units)

        # sum + non-linearity
        r_extra_flops = 2 * n_units
        # sum + non-linearity
        z_extra_flops = 2 * n_units
        # hadamard product + sum + non-linearity
        n_extra_flops = 3 * n_units
        h_extra_flops = 4 * n_units

        per_patch_gru_flops_by_layer.append(sum([
            x_affine_flops,
            h_affine_flops,
            r_extra_flops,
            z_extra_flops,
            n_extra_flops,
            h_extra_flops,
        ]))

    # Output layer

    per_patch_output_flops = retain_fraction * (2 * n_units * n_classes + n_classes)
    
    return FlopsInfo(
        patch_stride=patch_stride,
        per_sample_day_flops=per_sample_day_flops,
        per_patch_gru_flops_by_layer=tuple(per_patch_gru_flops_by_layer),
        per_patch_output_flops=per_patch_output_flops,
    )
