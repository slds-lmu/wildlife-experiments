"""Tuning with grid search."""
import math
import os
import time
from typing import Final, Dict
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
from wildlifeml.utils.io import load_json, load_pickle, save_as_json

from utils import product_dict


TIMESTR: Final[str] = time.strftime("%Y%m%d%H%M")


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

    # Fetch data
    dataset_is_train = load_pickle(
        os.path.join(cfg['data_dir'], 'dataset_is_train.pkl')
    )
    dataset_is_val = load_pickle(os.path.join(cfg['data_dir'], 'dataset_is_val.pkl'))
    empty_class_id = load_json(
        os.path.join(cfg['data_dir'], 'label_map.json')
    ).get('empty')

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

    # ----------------------------------------------------------------------------------
    # TUNING ---------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    # Define search grid
    search_space: Dict = {
        'model_backbone': ['xception', 'densenet121', 'inceptionresnetv2'],
        'finetune_layers': [0, 0.25, 0.5],
        'md_conf': [0.1, 0.5, 0.9]
    }
    search_grid = list(product_dict(**search_space))

    # Instantiate tuning archive
    tuning_archive: Dict = {}
    best_config: Dict = {}
    best_f1 = 0

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
        transfer_callbacks.append(WandbCallback(save_code=True, save_model=False))

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
        tf.random.set_seed(cfg['random_state'])
        trainer.fit(dataset_is_train, dataset_is_val_highconf)

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
        result = evaluator.evaluate(trainer)

        result.update(candidate)
        tuning_archive.update({f'iteration_{idx}': result})
        if result.get('f1') > best_f1:
            best_f1 = result.get('f1')
            candidate.update({'f1': best_f1})
            best_config.update(candidate)
        save_as_json(
            tuning_archive,
            os.path.join(cfg['result_dir'], f'{TIMESTR}_results_tuning_archive.json')
        )
        save_as_json(
            best_config,
            os.path.join(cfg['result_dir'], f'{TIMESTR}_results_tuning_best.json')
        )


if __name__ == '__main__':
    main()