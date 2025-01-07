import os
import torch
import logging
import argparse
import numpy as np
import pytorch_lightning as pl

from torch import nn
from models import FrameEncoder
from lightning_model import PLModel
from callback_loggers import get_callbacks
from training_utils import DirManager
from losses import *



def initialize_network(args):
    """
    Initializes the network.

    Args:
        args (Namespace): Parsed command-line arguments.
        frame_encoder (FrameEncoder): Initialized frame encoder.

    Returns:
        LinkSeg: Configured LinkSeg network.
    """
    network = FrameEncoder(
        n_mels=args.n_mels,
        conv_ndim=args.conv_ndim,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_embedding=args.n_embedding,
        f_min=args.f_min,
        f_max=args.f_max,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        attention_ndim=args.attention_ndim,
        attention_nlayers=args.attention_nlayers,
        attention_nheads=args.attention_nheads,
    )

    return network

def train(args):
    """
    Main training function for the PyTorch Lightning model.

    Args:
        args (Namespace): Parsed command-line arguments.
    """
    dir_manager = DirManager(output_dir=args.output_dir)

    # Initialize components
    network = initialize_network(args)
    print('#######', args.temperature_loss)
    loss_function = ContrastiveLoss(temperature=args.temperature_loss, n_conditions=args.n_conditions, symmetrical=args.symmetrical)
    #loss_function = SimCLR_2_full_Multi_Level()
    logging.info(f'Initialized network: {network}')

    model = PLModel(
        network=network,
        loss_function=loss_function,
        data_path=args.data_path,
        val_data_path=args.val_data_path,
        n_embedding=args.n_embedding,
        embedding_dim=args.embedding_dim,
        n_conditions=args.n_conditions,
        n_anchors=args.n_anchors,
        n_positives=args.n_positives,
        n_negatives=args.n_negatives,
        n_training_samples=args.n_training_samples,
        n_val_samples=args.n_val_samples,
        max_len=args.max_len,
        double_beats=args.double_beats,
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        learning_rate=args.learning_rate,
        optimizer=torch.optim.Adam,
        num_workers=args.num_workers,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        random_seed=args.random_seed,
    )

    # Set up callbacks and trainer
    callbacks = get_callbacks(
        patience=30, dir_manager=dir_manager, monitor='val_loss', mode='min'
    )
    num_gpus = torch.cuda.device_count()
    trainer = pl.Trainer(
        devices=num_gpus,
        num_nodes=args.num_nodes,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        accelerator="auto",
        gradient_clip_val=args.gradient_clip_val,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        accumulate_grad_batches=args.accumulate_grad_batches,
        enable_progress_bar=bool(args.enable_progress_bar),
    )

    # Start training
    trainer.fit(model)
    logging.info('Training completed.')

    # Save the best model
    logging.info('Loading and saving the best model.')
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = PLModel.load_from_checkpoint(best_model_path)
    torch.save(best_model.state_dict(), dir_manager.best_model_statedict)
    logging.info(f'Best model saved at: {dir_manager.best_model_statedict}')

def parse_args():
    """
    Parses command-line arguments.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Trainer args
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--gradient_clip_val', type=float, default=0.5)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--enable_progress_bar', type=int, default=1)

    # Model args
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--temperature_loss', type=float, default=.25)
    parser.add_argument('--symmetrical', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=42)

    # STFT args
    parser.add_argument('--n_mels', type=int, default=64)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--hop_length', type=int, default=256)
    parser.add_argument('--f_min', type=int, default=0)
    parser.add_argument('--f_max', type=int, default=11025)
    parser.add_argument('--sample_rate', type=int, default=22050)

    # Frame encoder args
    parser.add_argument('--n_embedding', type=int, default=64)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--conv_ndim', type=int, default=32)
    parser.add_argument('--attention_ndim', type=int, default=128)
    parser.add_argument('--attention_nheads', type=int, default=8)
    parser.add_argument('--attention_nlayers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Sampling args
    parser.add_argument('--n_anchors', type=int, default=16)
    parser.add_argument('--n_positives', type=int, default=16)
    parser.add_argument('--n_negatives', type=int, default=16)
    parser.add_argument('--n_conditions', type=int, default=2)
    parser.add_argument('--max_len', type=int, default=1500)
    parser.add_argument('--n_training_samples', type=int, default=1e6)
    parser.add_argument('--n_val_samples', type=int, default=1e6)
    parser.add_argument('--double_beats', type=int, default=2)

    # Paths
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--val_data_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./results/exp')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.info(f'Arguments: {args}')
    train(args)