
"""Prepare config file for experiments."""

import click
import os
from typing import Dict, Final
from wildlifeml.utils.io import save_as_json


@click.command()
@click.option(
    '--repo_dir', '-p', help='Your personal path to this repo.', required=True
)
@click.option('--root_dir', '-p', help='Path to original data root dir.', required=True)
@click.option(
    '--img_dir', '-p', help='Path to img dir under data root dir.', required=True
)
def main(repo_dir: str, root_dir: str, img_dir: str):
    cfg: Final[Dict] = {
        'root_dir': root_dir,
        'img_dir': img_dir,
        'data_dir': repo_dir + 'data/',
        'result_dir': repo_dir + 'results/',
        'label_file': 'labels.csv',
        'detector_file': 'images_megadetector.json',
        'mapping_file': 'bbox_map.json',
        'meta_file': 'stations.csv',
        'md_conf': 0.1,
        'md_batchsize': 32,
        'random_state': 123,
        'splits': (0.7, 0.15, 0.15),
        'batch_size': 32,
        'num_classes': 8,
        'transfer_epochs': 200,
        'finetune_epochs': 200,
        # 'finetune_layers': 2,
        'transfer_learning_rate': 0.001,
        'finetune_learning_rate': 0.001,
        'transfer_patience': 3,
        'finetune_patience': 3,
        'earlystop_metric': 'val_loss',
        # 'model_backbone': 'xception',
        'num_workers': 16,
        'eval_metrics': ['accuracy'],
        'test_logfile': 'test_logfile',
        'active_dir': repo_dir + 'active/',
        'al_batchsize': 32,
        'al_iterations': -1,
        'pretraining_ckpt': 'pretrained_weights',

    }
    save_as_json(cfg, os.path.join(repo_dir, 'configs/cfg.json'))


if __name__ == '__main__':
    main()