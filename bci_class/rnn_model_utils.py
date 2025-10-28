"""Utilities for dealing with the model."""
import os
import pathlib
from typing import Tuple

from omegaconf import OmegaConf
import torch

from model_training import rnn_model


def load_model(model_path: str, device: torch.device) -> Tuple['rnn_model.GRUDecoder', 'OmegaConf']:
    """Loads a given RNN model from disk."""
    model_args = OmegaConf.load(os.path.join(model_path, 'checkpoint/args.yaml'))

    # define model
    model = rnn_model.GRUDecoder(
        neural_dim=model_args['model']['n_input_features'],
        n_units=model_args['model']['n_units'], 
        n_days=len(model_args['dataset']['sessions']),
        n_classes=model_args['dataset']['n_classes'],
        rnn_dropout=model_args['model']['rnn_dropout'],
        input_dropout=model_args['model']['input_network']['input_layer_dropout'],
        n_layers=model_args['model']['n_layers'],
        patch_size=model_args['model']['patch_size'],
        patch_stride=model_args['model']['patch_stride'],
    )

    # load model weights
    checkpoint = torch.load(os.path.join(model_path, 'checkpoint/best_checkpoint'), weights_only=False, map_location=device)
    # rename keys to not start with "module." (happens if model was saved with DataParallel)
    for key in list(checkpoint['model_state_dict'].keys()):
        checkpoint['model_state_dict'][key.replace("module.", "")] = checkpoint['model_state_dict'].pop(key)
        checkpoint['model_state_dict'][key.replace("_orig_mod.", "")] = checkpoint['model_state_dict'].pop(key)
    model.load_state_dict(checkpoint['model_state_dict'])

    # add model to device
    model.to(device) 

    # set model to eval mode
    model.eval()

    return model, model_args


def save_model(model: 'rnn_model.GRUDecoder', model_args: 'OmegaConf', model_path: str) -> 'rnn_model.GRUDecoder':
    """Saves a model in a way that will be compatible with load_model.

    If model_path exists, then will just write into it. Its parent directory must exist.
    """

    # Make the directories that we'll write it if they don't already exist.
    pathlib.Path(model_path).mkdir(parents=False, exist_ok=True) 
    pathlib.Path(os.path.join(model_path, 'checkpoint')).mkdir(parents=False, exist_ok=True) 

    # Save the checkpoint.
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    torch.save(checkpoint, os.path.join(model_path, 'checkpoint/best_checkpoint'))

    # Save the arguments.
    with open(os.path.join(model_path, 'checkpoint/args.yaml'), 'w') as f:
        OmegaConf.save(config=model_args, f=f)
