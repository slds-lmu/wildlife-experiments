"""In-sample results."""

import time
import click
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import gc
from typing import Dict, Final, List
from wildlifeml.data import subset_dataset
from wildlifeml.training.trainer import WildlifeTrainer
from wildlifeml.training.evaluator import Evaluator
from wildlifeml.utils.datasets import (
    separate_empties,
    map_bbox_to_img,
)
from wildlifeml.utils.io import (
    load_json,
    load_pickle,
)
from wildlifeml.utils.misc import flatten_list
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

timestr = time.strftime("%Y%m%d%H%M")


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

    # Prepare training
    dataset_is_train = load_pickle(
        os.path.join(cfg['data_dir'], 'dataset_is_train.pkl')
    )
    dataset_is_val = load_pickle(
        os.path.join(cfg['data_dir'], 'dataset_is_val.pkl')
    )
    dataset_is_trainval = load_pickle(os.path.join(
        cfg['data_dir'], 'dataset_is_trainval.pkl')
    )

    empty_class_id = load_json(
        os.path.join(cfg['data_dir'], 'label_map.json')
    ).get('empty')

    df = pd.read_csv(os.path.join(repo_dir, 'data/tune.csv'))

    keys_train = list(
        set([map_bbox_to_img(k) for k in dataset_is_train.keys])
    )
    keys_val = list(
        set([map_bbox_to_img(k) for k in dataset_is_val.keys])
    )
    for index, row in df.iterrows():
        md_thresh = row.md_thresh
        batch_size = row.batch_size
        model_backbone = row.model_backbone
        transfer_learning_rate = row.transfer_learning_rate
        finetune_learning_rate = row.finetune_learning_rate
        transfer_epochs = row.transfer_epochs
        finetune_epochs = row.finetune_epochs
        finetune_layers = row.finetune_layers
        transfer_patience = row.transfer_patience
        finetune_patience = row.finetune_patience

        if transfer_epochs > 0:
            transfer_callbacks = [
                EarlyStopping(
                    monitor=cfg['earlystop_metric'],
                    patience=transfer_patience,
                )
            ]
        else:
            transfer_callbacks = None
        if finetune_epochs > 0:
            finetune_callbacks = [
                EarlyStopping(
                    monitor=cfg['earlystop_metric'],
                    patience=finetune_patience,
                )
            ]
        else:
            finetune_callbacks = None

        trainer_args: Dict = {
            'batch_size': batch_size,
            'loss_func': keras.losses.SparseCategoricalCrossentropy(),
            'num_classes': cfg['num_classes'],
            'transfer_epochs': transfer_epochs,
            'finetune_epochs': finetune_epochs,
            'transfer_optimizer': Adam(transfer_learning_rate),
            'finetune_optimizer': Adam(finetune_learning_rate),
            'finetune_layers': finetune_layers,
            'model_backbone': model_backbone,
            'transfer_callbacks': transfer_callbacks,
            'finetune_callbacks': finetune_callbacks,
            'num_workers': cfg['num_workers'],
            'eval_metrics': cfg['eval_metrics'],
        }
        _, keys_nonempty_bbox = separate_empties(
            os.path.join(cfg['data_dir'], cfg['detector_file']), float(md_thresh)
        )
        keys_nonempty_bbox = list(
            set(keys_nonempty_bbox).intersection(set(dataset_is_trainval.keys))
        )
        dataset_thresh = subset_dataset(
            dataset_is_trainval,
            keys_nonempty_bbox
        )
        dataset_train_thresh = subset_dataset(
            dataset_thresh,
            flatten_list([dataset_thresh.mapping_dict[k] for k in keys_train])
        )
        dataset_val_thresh = subset_dataset(
            dataset_thresh,
            flatten_list([dataset_thresh.mapping_dict[k] for k in keys_val])
        )
        trainer = WildlifeTrainer(**trainer_args)
        tf.random.set_seed(cfg['random_state'])
        trainer.fit(
            train_dataset=dataset_train_thresh,
            val_dataset=dataset_val_thresh
        )

        evaluator = Evaluator(
            label_file_path=os.path.join(cfg['data_dir'], cfg['label_file']),
            detector_file_path=os.path.join(cfg['data_dir'], cfg['detector_file']),
            dataset=dataset_is_val,
            num_classes=cfg['num_classes'],
            empty_class_id=empty_class_id,
            conf_threshold=float(md_thresh),
        )

        results = evaluator.evaluate(trainer)

        for metric in ['acc', 'prec', 'rec', 'f1']:
            df.loc[index, metric] = results.get(metric)

        df.to_csv(
            os.path.join(cfg['result_dir'], f'{timestr}_tune.csv'),
            index=False,
            )
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    main()
