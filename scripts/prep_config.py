"""Prepare config file fr experiments."""

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
        'batch_size': 128,
        'num_classes': 8,
        'transfer_epochs': 1,
        'finetune_epochs': 1,
        'finetune_layers': 1,
        'transfer_learning_rate': 1e-3,
        'finetune_learning_rate': 1e-3,
        'model_backbone': 'resnet50',
        'num_workers': 16,
        'eval_metrics': ['accuracy'],
        'test_logfile': 'test_logfile',
        'active_dir': repo_dir + 'active/',
        'al_batchsize': 32,
        'al_iterations': 1,
        'pretraining_ckpt': 'pretrained_weights',
        'n_trials': 5,
        'time_budget': 360000, #100hours
        'max_concurrent_trials': 4,
        'search_alg_id': 'randomsearch',
        'scheduler_alg_id': 'ashascheduler',
    }
    save_as_json(cfg, os.path.join(repo_dir, 'configs/cfg.json'))


if __name__ == '__main__':
    main()