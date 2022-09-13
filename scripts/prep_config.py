"""Prepare config file fr experiments."""

import os
from typing import Dict, Final
from wildlifeml.utils.io import save_as_json

PATH: Final[str] = '/home/wimmerl/projects/wildlife-experiments/'
CFG: Final[Dict] = {
    'root_dir': '/common/bothmannl/',
    'img_dir': 'wildlife_images/usecase2/original_images/',
    'data_dir': PATH + 'data/',
    'result_dir': PATH + 'results/',
    'label_file': 'labels.csv',
    'detector_file': 'images_megadetector.json',
    'mapping_file': 'bbox_map.json',
    'info_file': 'metadata/uc2_labels.csv',
    'meta_file': 'stations.csv',
    'md_conf': 0.1,
    'md_batchsize': 32,
    'random_state': 123,
    'splits': (0.7, 0.15, 0.15),
    'batch_size': 128,
    'num_classes': 8,
    'transfer_epochs': 0,
    'finetune_epochs': 0,
    'finetune_layers': 1,
    'model_backbone': 'resnet50',
    'num_workers': 16,
    'eval_metrics': ['accuracy'],
    'test_logfile': 'test_logfile',
    'active_dir': PATH + 'active/',
    'al_batchsize': 32,
    'al_iterations': 1,
    'pretraining_ckpt': 'pretrained_weights',
}
save_as_json(CFG, os.path.join(PATH, 'configs/cfg.json'))