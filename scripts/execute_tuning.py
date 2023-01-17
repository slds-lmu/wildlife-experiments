"""Tuning with grid search."""
import os
from typing import Final, Dict
import click
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from wildlifeml.data import subset_dataset
from wildlifeml.training.evaluator import Evaluator
from wildlifeml.training.models import ModelFactory
from wildlifeml.training.trainer import WildlifeTrainer
from wildlifeml.utils.datasets import separate_empties
from wildlifeml.utils.io import load_json, load_pickle

from utils import product_dict


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
    dataset_is_val = load_pickle(
        os.path.join(cfg['data_dir'], 'dataset_is_val.pkl')
    )
    dataset_is_test = load_pickle(
        os.path.join(cfg['data_dir'], 'dataset_is_test.pkl')
    )
    empty_class_id = load_json(
        os.path.join(cfg['data_dir'], 'label_map.json')
    ).get('empty')

    # Define early-stopping callbacks
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

    # Define evaluator
    evaluator = Evaluator(
        label_file_path=os.path.join(cfg['data_dir'], cfg['label_file']),
        detector_file_path=os.path.join(cfg['data_dir'], cfg['detector_file']),
        dataset=dataset_is_test,
        num_classes=cfg['num_classes'],
        empty_class_id=empty_class_id,
    )

    # ----------------------------------------------------------------------------------
    # TUNING ---------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    # Define search grid
    search_space: Dict = {
        'model_backbone': ['xception', 'densenet121', 'inceptionresnetv2'],
        'finetune_layers': ['last', 'half', 'all'],
        'md_conf': [0.1, 0.5, 0.9]
    }
    search_grid = list(product_dict(**search_space))

    # Instantiate tuning archive
    tuning_archive: Dict = {}
    best_config: Dict = {}
    best_f1 = 0

    for idx, candidate in enumerate(search_grid):

        # Remove empty images from train/val keys according to MD threshold
        _, keys_all_nonempty = separate_empties(
            detector_file_path=os.path.join(cfg['data_dir'], cfg['detector_file']),
            conf_threshold=candidate['md_conf']
        )
        keys_is_train = list(
            set(dataset_is_train.keys).intersection(set(keys_all_nonempty))
        )
        keys_is_val = list(
            set(dataset_is_val.keys).intersection(set(keys_all_nonempty))
        )
        dataset_is_train = subset_dataset(dataset_is_train, keys_is_train)
        dataset_is_val = subset_dataset(dataset_is_val, keys_is_val)

        # Determine number of finetuning layers
        model = ModelFactory.get(
            model_id=candidate['model_backbone'], num_classes=cfg['num_classes']
        )
        breakpoint()

        # Define trainer args
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

        trainer = WildlifeTrainer(**trainer_args)
        print(f'---> Training with configuration {idx}')
        tf.random.set_seed(cfg['random_state'])
        trainer.fit(dataset_is_train, dataset_is_val)
        print(f'---> Evaluating for configuration {idx}')
        result = evaluator.evaluate(trainer)

        tuning_archive.update({f'iteration_{idx}': result})
        if result.get('f1') > best_f1:
            best_f1 = result.get('f1')
            best_config.update(candidate)


if __name__ == '__main__':
    main()