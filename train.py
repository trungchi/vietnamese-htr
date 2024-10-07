from model import ModelTF  # Import only the required model
import pytorch_lightning as pl
import argparse
from config import Config, initialize
from typing import Any
import os
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    # parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=1)
    #
    # # Add trainer options to parser
    # parser.add_argument('--max_epochs', type=int, default=10, help='Maximum number of epochs to train.')
    # parser.add_argument('--gpus', type=int, default=0, help='Number of GPUs to use. Set to 0 for CPU.')
    # parser.add_argument('--deterministic', type=bool, default=False, help='Set to True for deterministic training.')
    #
    # # Determine which model to use
    # parser.add_argument('model_name', type=str, choices=['tf', 'rnn', 'ctc'], help='Model type: Transformer (tf), RNN (rnn), or CTC (ctc)')
    # parser.add_argument('config_path', type=str, help='Path to the configuration file.')

    # Parse known arguments to decide model-specific arguments
    temp_args, _ = parser.parse_known_args()
    temp_args.model_name = 'tf'
    # Add model-specific arguments based on the selected model
    Model: Any
    if temp_args.model_name == 'tf':
        parser = ModelTF.add_model_specific_args(parser)
        Model = ModelTF
    # Uncomment and implement the following if you have ModelRNN and ModelCTC defined
    # elif temp_args.model_name == 'rnn':
    #     parser = ModelRNN.add_model_specific_args(parser)
    #     Model = ModelRNN
    # elif temp_args.model_name == 'ctc':
    #     parser = ModelCTC.add_model_specific_args(parser)
    #     Model = ModelCTC

    args = parser.parse_args()

    # Set up checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',  # Directory to save checkpoints
        filename='{epoch:02d}-{val_loss:.2f}',  # Custom filename pattern
        save_top_k=1,
        verbose=True,
        monitor='val_loss',  # Changed to 'val_loss' for a more common metric
        mode='min'
    )

    # Determine if GPU can be used
    use_gpu = torch.cuda.is_available() and args.gpus > 0
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu" if use_gpu else "cpu",  # Use GPU if available
        devices=args.gpus if use_gpu else 1,      # Set number of devices (GPUs or CPUs)
        deterministic=True,           # Ensure reproducibility
        callbacks=[checkpoint_callback],
    )

    # Load configuration from file and arguments
    args.config_path = 'config/base.yaml'
    args.seed = 9498
    dict_args = vars(args)
    config = Config(dict_args.pop('config_path'), **dict_args).config

    # Seed for reproducibility
    pl.seed_everything(dict_args['seed'])

    # Initialize model
    model = Model(config)

    # Start training
    trainer.fit(model)
    print("Training finished")
