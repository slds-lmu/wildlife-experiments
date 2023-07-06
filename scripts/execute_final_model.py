"""In-sample results."""

import time

import click
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import gc
from typing import Dict, Final, List
from wildlifeml.data import subset_dataset
from wildlifeml.training.trainer import WildlifeTrainer
from wildlifeml.training.active import ActiveLearner
from wildlifeml.training.evaluator import Evaluator
from wildlifeml.utils.datasets import separate_empties, map_bbox_to_img, do_stratified_splitting
from wildlifeml.utils.io import (
    load_csv,
    load_json,
    load_pickle,
    save_as_csv,
    save_as_pickle,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from utils import seed_everything, MyEarlyStopping

THRESH_TUNED: Final[float] = 0.1
BACKBONE_TUNED: Final[str] = 'xception'
FTLAYERS_TUNED: Final[int] = 0


@click.command()
@click.option(
    '--repo_dir', '-p', help='Your personal path to this repo.', required=True
)
@click.option(
    '--random_seed', '-s', help='Random seed.', required=True
)
def main(repo_dir: str, random_seed: int):

    cfg: Final[Dict] = load_json(os.path.join(repo_dir, 'configs/cfg.json'))
    seed_everything(random_seed)
    os.makedirs(os.path.join(cfg['result_dir'], 'final'), exist_ok=True)

    # Get data
    stations_dict = {
        k: {'station': v}
        for k, v in load_csv(os.path.join(cfg['data_dir'], 'stations.csv'))
    }
    dataset_train = load_pickle(os.path.join(cfg['data_dir'], 'dataset_full_train.pkl'))
    dataset_val = load_pickle(os.path.join(cfg['data_dir'], 'dataset_full_val.pkl'))
    _, keys_all_nonempty = separate_empties(
        os.path.join(cfg['data_dir'], cfg['detector_file']), float(THRESH_TUNED)
    )
    keys_train = list(
        set(dataset_train.keys).intersection(set(keys_all_nonempty))
    )
    keys_val = list(
        set(dataset_val.keys).intersection(set(keys_all_nonempty))
    )
    dataset_train = subset_dataset(dataset_train, keys_train)
    dataset_val = subset_dataset(dataset_val, keys_val)
    dataset_val.shuffle = False
    dataset_val.augmentation = None

    # Prepare training
    trainer_args: Final[Dict] = {
        'batch_size': cfg['batch_size'],
        'loss_func': keras.losses.SparseCategoricalCrossentropy(),
        'num_classes': cfg['num_classes'],
        'transfer_epochs': cfg['transfer_epochs'],
        'finetune_epochs': cfg['finetune_epochs'],
        'finetune_layers': FTLAYERS_TUNED,
        'model_backbone': BACKBONE_TUNED,
        'num_workers': cfg['num_workers'],
        'eval_metrics': cfg['eval_metrics'],
    }
    transfer_callbacks = [
        EarlyStopping(
            monitor=cfg['earlystop_metric'],
            patience=2 * cfg['transfer_patience'],
        ),
        ReduceLROnPlateau(
            monitor=cfg['earlystop_metric'],
            patience=cfg['transfer_patience'],
            factor=0.1,
            verbose=1,
        ),
    ]
    finetune_callbacks = [
        EarlyStopping(
            monitor=cfg['earlystop_metric'],
            patience=2 * cfg['finetune_patience'],
        ),
        ReduceLROnPlateau(
            monitor=cfg['earlystop_metric'],
            patience=cfg['finetune_patience'],
            factor=0.1,
            verbose=1,
        )
    ]
    ckpt_dir = os.path.join(cfg['data_dir'], 'final_ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_callback = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, 'ckpt_final.hdf5'),
            monitor=cfg['earlystop_metric'],
            mode='min',
            save_weights_only=False,
            save_best_only=True,
        )
    ]
    if FTLAYERS_TUNED == 0 or cfg['finetune_epochs'] == 0:
        transfer_callbacks.append(ckpt_callback)
    else:
        finetune_callbacks.append(ckpt_callback)
    this_trainer_args: Dict = dict(
        {
            'transfer_callbacks': transfer_callbacks,
            'finetune_callbacks': finetune_callbacks,
            'transfer_optimizer': Adam(cfg['transfer_learning_rate']),
            'finetune_optimizer': Adam(cfg['finetune_learning_rate']),
        },
        **trainer_args
    )

    # Train
    trainer = WildlifeTrainer(**this_trainer_args)
    print('---> Training final model')
    trainer.fit(train_dataset=dataset_train, val_dataset=dataset_val)


if __name__ == '__main__':
    main()
