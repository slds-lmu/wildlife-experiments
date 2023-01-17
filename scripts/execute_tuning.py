"""Tuning with grid search."""
import itertools
import random
import time
from typing import Final, Dict

import click
from copy import deepcopy
import os

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from wildlifeml.utils.io import load_json, load_csv, load_pickle


@click.command()
@click.option(
    '--repo_dir', '-p', help='Your personal path to this repo.', required=True
)
def main(repo_dir: str):

    # ----------------------------------------------------------------------------------
    # GLOBAL ---------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    cfg: Final[Dict] = load_json(os.path.join(repo_dir, 'configs/cfg.json'))
    os.makedirs(cfg['result_dir'], exist_ok=True)

    # Get metadata
    label_dict = {
        k: v
        for k, v in load_csv(os.path.join(cfg['data_dir'], cfg['label_file']))
    }
    stations_dict = {
        k: {'station': v}
        for k, v in load_csv(os.path.join(cfg['data_dir'], 'stations.csv'))
    }

    # Prepare training
    dataset_is_train = load_pickle(
        os.path.join(cfg['data_dir'], 'dataset_is_train.pkl')
    )

    # Remove empty images from train/val keys (training is only based on non-empty
    # images, while evaluation on test takes all keys into account)
    # _, keys_all_nonempty = separate_empties(
    #     detector_file_path=os.path.join(cfg['data_dir'], cfg['detector_file']),
    #     conf_threshold=cfg['md_conf']
    # )
    # keys_is_train = list(set(keys_is_train).intersection(set(keys_all_nonempty)))
    # keys_is_val = list(set(keys_is_val).intersection(set(keys_all_nonempty)))
    # keys_oos_train = list(set(keys_oos_train).intersection(set(keys_all_nonempty)))
    # keys_oos_val = list(set(keys_oos_val).intersection(set(keys_all_nonempty)))

    transfer_callbacks = [
        EarlyStopping(
            monitor=cfg['earlystop_metric'],
            patience=cfg['transfer_patience'],
        )
    ]

    finetune_callbacks = [
        EarlyStopping(
            monitor=cfg['earlystop_metric'],
            patience=cfg['finetune_patience'],
        )
    ]

    trainer_args: Dict = {
        'batch_size': cfg['batch_size'],
        'loss_func': keras.losses.SparseCategoricalCrossentropy(),
        'num_classes': cfg['num_classes'],
        'transfer_epochs': cfg['transfer_epochs'],
        'finetune_epochs': cfg['finetune_epochs'],
        'transfer_optimizer': Adam(cfg['transfer_learning_rate']),
        'finetune_optimizer': Adam(cfg['finetune_learning_rate']),
        'finetune_layers': cfg['finetune_layers'],
        'model_backbone': cfg['model_backbone'],
        'transfer_callbacks': transfer_callbacks,
        'finetune_callbacks': finetune_callbacks,
        'num_workers': cfg['num_workers'],
        'eval_metrics': cfg['eval_metrics'],
    }
    empty_class_id = load_json(
        os.path.join(cfg['data_dir'], 'label_map.json')
    ).get('empty')

    evaluator_is = Evaluator(
        label_file_path=os.path.join(cfg['data_dir'], cfg['label_file']),
        detector_file_path=os.path.join(cfg['data_dir'], cfg['detector_file']),
        dataset=dataset_is_test,
        num_classes=cfg['num_classes'],
        empty_class_id=empty_class_id,
    )
    evaluator_oos = Evaluator(
        label_file_path=os.path.join(cfg['data_dir'], cfg['label_file']),
        detector_file_path=os.path.join(cfg['data_dir'], cfg['detector_file']),
        dataset=dataset_oos_test,
        num_classes=cfg['num_classes'],
        empty_class_id=empty_class_id,
    )

    # ----------------------------------------------------------------------------------
    # TUNING ---------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    search_space: Dict = {
        'model_backbone': ['xception', 'densenet121', 'inceptionresnetv2'],
        'finetune_layers': ['last', 'half', 'all'],
        'md_threshold': [0.1, 0.5, 0.9]
    }
    search_grid = list(product_dict(**search_space))

    for candidate in search_grid:
        trainer_args: Dict = {
            'batch_size': cfg['batch_size'],
            'loss_func': keras.losses.SparseCategoricalCrossentropy(),
            'num_classes': cfg['num_classes'],
            'transfer_epochs': cfg['transfer_epochs'],
            'finetune_epochs': cfg['finetune_epochs'],
            'transfer_optimizer': Adam(cfg['transfer_learning_rate']),
            'finetune_optimizer': Adam(cfg['finetune_learning_rate']),
            'finetune_layers': candidate['finetune_layers'],
            'model_backbone': candidate['model_backbone'],
            'transfer_callbacks': transfer_callbacks,
            'finetune_callbacks': finetune_callbacks,
            'num_workers': cfg['num_workers'],
            'eval_metrics': cfg['eval_metrics'],
        }