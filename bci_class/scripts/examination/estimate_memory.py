"""Estimates flops given some configuration of a model."""
import argparse

from bci_class.flops import memory_estimators

###############################################################################
parser = argparse.ArgumentParser(description='Evaluate a pretrained RNN model on the copy task dataset.')

parser.add_argument('--neural_dim', type=int, help='number of channels in a single timestep (e.g. 512)')
parser.add_argument('--n_units', type=int, help='number of hidden units in each recurrent layer - equal to the size of the hidden state')
parser.add_argument('--patch_size', type=int, help='the number of timesteps to concat on initial input layer')
parser.add_argument('--patch_stride', type=int, help='the number of timesteps to stride over when concatenating initial input ')
parser.add_argument('--n_layers', type=int, help='number of recurrent layers ')
parser.add_argument('--n_classes', type=int, help='number of classes ')


parser.add_argument('--bytes_per_parameter', type=int, help='number of bytes per parameter.')


# Reduced rank flags. Use these when the saved model is reduced rank parameterization.
parser.add_argument('--reduced_rank', type=bool, default=False,
                    help='Whether the saved model is a reduced rank model.')
parser.add_argument('--day_weights_rank', type=int, help='Rank to compress day_weights to. Leave unset or set to 0 to not compress them.')
parser.add_argument('--gru_rank', type=int, help='Rank to compress gru weights to. Leave unset or set to 0 to not compress them.')
parser.add_argument('--gru_rank_ih_l0', type=int, help='Rank to compress weight_ih_l0 to. Only has an effect if gru_rank is not zero/None. If zero/None, equivalent to -gru_rank')

# Magnitude pruning flags. Use these when the saved model has magnitufe pruning parameterization.
parser.add_argument('--magnitude_pruned', type=bool, default=False,
                    help='Whether the saved model is a magnitude pruned model.')
parser.add_argument('--retain_fraction', type=float, help='If magnitude pruned, this is the fraction of weights that are retained.')

args = parser.parse_args()

###############################################################################


common_kwargs = {
    'neural_dim': args.neural_dim,
    'n_units': args.n_units,
    'patch_size': args.patch_size,
    'n_layers': args.n_layers,
    'n_classes': args.n_classes,
}

if args.reduced_rank:
    n_params = memory_estimators.compute_parameter_count_rank_reduced(
        **common_kwargs,
        day_weights_rank=args.day_weights_rank,
        gru_rank=args.gru_rank,
        gru_rank_ih_l0=args.gru_rank_ih_l0,
    )

elif args.magnitude_pruned:
    n_params = memory_estimators.compute_parameter_count_original_model(
        **common_kwargs,
    )
    # assumes index size is same as parameter size.
    n_params *= 2 * args.retain_fraction
else:
    n_params = memory_estimators.compute_parameter_count_original_model(**common_kwargs)


print(f'bytes: {n_params * args.bytes_per_parameter}')
