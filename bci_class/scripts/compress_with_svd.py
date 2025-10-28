"""Compresses an RNN model with SVD.

NOTE: The actual parameters will not get compressed, i.e. they will get stored as full rank.
"""
import os

import argparse
import torch

from bci_class import rnn_model_utils
from bci_class.compression import svd_compression

###############################################################################
parser = argparse.ArgumentParser(description='Compresses an RNN model with SVD.')

parser.add_argument('--input_path', type=str, help='Path to the input pretrained model directory.')
parser.add_argument('--output_path', type=str, help='Path to the output pretrained model directory.')

parser.add_argument('--day_weights_rank', type=int, help='Rank to compress day_weights to. Leave unset or set to 0 to not compress them.')
parser.add_argument('--gru_rank', type=int, help='Rank to compress gru weights to. Leave unset or set to 0 to not compress them.')
parser.add_argument('--gru_rank_ih_l0', type=int, help='Rank to compress weight_ih_l0 to. Only has an effect if gru_rank is not zero/None. If zero/None, equivalent to -gru_rank')

args = parser.parse_args()
###############################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    model, model_args = rnn_model_utils.load_model(args.input_path, device)
    print('Model loaded.')

    if args.day_weights_rank:
        svd_compression.compress_day_weights_via_svd(model, args.day_weights_rank)

    if args.gru_rank:
        svd_compression.compress_gru_via_svd(model, args.gru_rank, args.gru_rank_ih_l0)

    print('Saving model...')
    rnn_model_utils.save_model(model, model_args, args.output_path)
