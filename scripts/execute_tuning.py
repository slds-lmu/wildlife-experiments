"""Tuning with grid search."""
import math
import os
import gc
import time

import numpy as np
import pandas as pd
from typing import Final, Dict, List
import click
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import wandb
from wandb.keras import WandbCallback
from wildlifeml.data import subset_dataset
from wildlifeml.training.evaluator import Evaluator
from wildlifeml.training.models import ModelFactory
from wildlifeml.training.trainer import WildlifeTrainer
from wildlifeml.utils.datasets import separate_empties
from wildlifeml.utils.io import load_json, load_pickle

from utils import product_dict, seed_everything


TIMESTR: Final[str] = time.strftime("%Y%m%d%H%M")


@click.command()
@click.option(
    '--repo_dir', '-p', help='Your personal path to this repo.', required=True
)
@click.option(
    '--random_seed', '-s', help='Random seed.', required=True
)
def main(repo_dir: str, random_seed: int):

    # ----------------------------------------------------------------------------------
    # GLOBAL ---------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    seed_everything(random_seed)
    cfg: Final[Dict] = load_json(os.path.join(repo_dir, 'configs/cfg.json'))
    os.makedirs(cfg['result_dir'], exist_ok=True)

    # Fetch data
    dataset_is_train = load_pickle(
        os.path.join(cfg['data_dir'], 'dataset_is_train.pkl')
    )
    dataset_is_val = load_pickle(os.path.join(cfg['data_dir'], 'dataset_is_val.pkl'))
    empty_class_id = load_json(
        os.path.join(cfg['data_dir'], 'label_map.json')
    ).get('empty')

    # ----------------------------------------------------------------------------------
    # TUNING ---------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    # Define search grid
    search_space: Dict = {
        'model_backbone': ['resnet50'],  # ['xception', 'densenet121', 'inception_resnet_v2'],
        'finetune_layers': [0.],  # [0, 0.05, 0.25, 0.5],
        'md_conf': [0.3, 0.5, 0.7, 0.9]  # np.arange(0.1, 1, 0.2).round(2).tolist()
    }
    search_grid = list(product_dict(**search_space))

    # Instantiate tuning archive
    tuning_archive: List = []
    col_names: Final[List] = [
        'ts',
        'md_threshold',
        'backbone',
        'finetune_layers',
        'f1',
        'acc',
        'prec',
        'rec',
        'empty_tnr',
        'empty_tpr',
        'empty_fnr',
        'empty_fpr',
    ]

    for idx, candidate in enumerate(search_grid):

        this_conf = candidate['md_conf']
        this_backbone = candidate['model_backbone']

        # Remove empty images from train/val keys according to MD threshold
        _, keys_all_nonempty = separate_empties(
            detector_file_path=os.path.join(cfg['data_dir'], cfg['detector_file']),
            conf_threshold=this_conf
        )
        keys_is_train = list(
            set(dataset_is_train.keys).intersection(set(keys_all_nonempty))
        )
        keys_is_val = list(
            set(dataset_is_val.keys).intersection(set(keys_all_nonempty))
        )
        dataset_is_train = subset_dataset(dataset_is_train, keys_is_train)
        dataset_is_val_highconf = subset_dataset(dataset_is_val, keys_is_val)
        dataset_is_val.shuffle = False
        dataset_is_val_highconf.shuffle = False
        dataset_is_val.augmentation = None
        dataset_is_val_highconf.augmentation = None

        # Determine number of finetuning layers
        model = ModelFactory.get(model_id=this_backbone, num_classes=cfg['num_classes'])
        n_layers_featext = len(model.get_layer(this_backbone).layers)
        this_finetune_layers = math.floor(
            candidate['finetune_layers'] * n_layers_featext
        )

        # Add logging
        wandb.init(
            project='wildlilfe',
            tags=[
                f'conf_{this_conf}',
                this_backbone,
                f'ftlayers_{this_finetune_layers}'
            ]
        )

        # Define early-stopping callbacks
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
            WandbCallback(save_code=True, save_model=False),
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

        # Define trainer args
        trainer_args: Dict = {
            'batch_size': cfg['batch_size'],
            'loss_func': keras.losses.SparseCategoricalCrossentropy(),
            'num_classes': cfg['num_classes'],
            'transfer_epochs': cfg['transfer_epochs'],
            'finetune_epochs': cfg['finetune_epochs'],
            'transfer_optimizer': Adam(cfg['transfer_learning_rate']),
            'finetune_optimizer': Adam(cfg['finetune_learning_rate']),
            'finetune_layers': this_finetune_layers,
            'model_backbone': this_backbone,
            'transfer_callbacks': transfer_callbacks,
            'finetune_callbacks': finetune_callbacks,
            'num_workers': cfg['num_workers'],
            'eval_metrics': cfg['eval_metrics'],
        }

        # Train
        trainer = WildlifeTrainer(**trainer_args)
        print(f'---> Training with configuration {idx}')
        trainer.fit(dataset_is_train, dataset_is_val_highconf)
        wandb.finish()

        # Define evaluator (everything below candidate['md_conf'] is treated as filtered
        # by the MD, the rest is predicted by the trainer)
        evaluator = Evaluator(
            label_file_path=os.path.join(cfg['data_dir'], cfg['label_file']),
            detector_file_path=os.path.join(cfg['data_dir'], cfg['detector_file']),
            dataset=dataset_is_val,
            num_classes=cfg['num_classes'],
            empty_class_id=empty_class_id,
            conf_threshold=this_conf
        )
        print(f'---> Evaluating for configuration {idx}')
        evaluator.evaluate(trainer)
        result = evaluator.compute_metrics()
        tuning_archive.append(
            [
                TIMESTR,
                this_conf,
                this_backbone,
                this_finetune_layers,
                result['f1'],
                result['acc'],
                result['prec'],
                result['rec'],
                result['conf_empty']['tnr'],
                result['conf_empty']['tpr'],
                result['conf_empty']['fnr'],
                result['conf_empty']['fpr'],
            ]
        )
        df = pd.DataFrame(tuning_archive, columns=col_names)
        archive_file = os.path.join(
            cfg['result_dir'], f'results_tuning_archive_{random_seed}.csv'
        )
        if os.path.exists(archive_file):
            existing = pd.read_csv(archive_file, usecols=col_names)
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df
        combined = combined.drop_duplicates(subset=list(set(col_names) - {'ts'}))
        combined.to_csv(archive_file)

        tf.keras.backend.clear_session()
        gc.collect()


if __name__ == '__main__':
    main()
