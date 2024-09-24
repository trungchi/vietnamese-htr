from model import ModelTF, ModelRNN, ModelCTC
import pytorch_lightning as pl
import argparse
from config import Config, initialize
from typing import Any
import os
from pytorch_lightning.callbacks import ModelCheckpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)

    # Add trainer options to parser
    # parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum number of epochs to train.')
    parser.add_argument('--gpus', type=int, default=0, help='Number of GPUs to use. Set to 0 for CPU.')
    parser.add_argument('--deterministic', type=bool, default=False, help='Set to True for deterministic training.')


    # figure out which model to use
    parser.add_argument('model_name', type=str, choices=['tf', 'rnn', 'ctc'], help='Transformer or RNN or CTC')
    parser.add_argument('config_path', type=str)

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # Add model options to parser
    Model: Any
    if temp_args.model_name == 'tf':
        parser = ModelTF.add_model_specific_args(parser)
        Model = ModelTF
    elif temp_args.model_name == 'rnn':
        parser = ModelRNN.add_model_specific_args(parser)
        Model = ModelRNN
    elif temp_args.model_name == 'ctc':
        parser = ModelCTC.add_model_specific_args(parser)
        Model = ModelCTC

    args = parser.parse_args()

    # checkpoint_callback = ModelCheckpoint(
    #     filepath=None,
    #     save_top_k=1,
    #     verbose=True,
    #     monitor='WER',
    #     mode='min',
    #     prefix=''
    # )
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',  # Specify the directory for saving checkpoints
        filename='{epoch:02d}-{val_loss:.2f}',  # Use a custom filename pattern
        save_top_k=1,
        verbose=True,
        monitor='WER',
        mode='min'
    )


    # trainer = pl.Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,  # Make sure to define this argument in your parser
        accelerator='gpu',           # Specify using GPU
        devices=args.gpus,          # Number of GPUs to use
        deterministic=args.deterministic,  # Ensure this argument exists in your parser
        callbacks=[checkpoint_callback],
    )

    dict_args = vars(args)

    config = Config(dict_args.pop('config_path'), **dict_args).config

    # pl.seed_everything(dict_args['seed'])
    # cnn = initialize(config['cnn'])

    # pl.seed_everything(dict_args['seed'])
    # vocab = initialize(config['vocab'], add_blank=False)

    pl.seed_everything(dict_args['seed'])
    model = Model(config)

    pl.seed_everything(dict_args['seed'])
    trainer.fit(model)
