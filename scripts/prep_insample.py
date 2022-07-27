"""Prepare objects shared across in-sample experiments."""

import albumentations as A
import os
from tensorflow import keras
from typing import Final, Dict

from wildlifeml.training.trainer import WildlifeTrainer
from wildlifeml.training.evaluator import Evaluator
from wildlifeml.data import WildlifeDataset, modify_dataset
from wildlifeml.utils.datasets import do_stratified_splitting, do_stratified_cv
from wildlifeml.utils.io import load_csv, load_csv_dict, load_json

CFG: Final[Dict] = {
    'img_dir': '/common/bothmannl/wildlife_images/usecase2/original_images/',
    'root_dir': '/home/wimmerl/projects/wildlife-experiments/data/',
    'label_file': 'labels.csv',
    'detector_file': 'images_megadetector.json',
    'mapping_file': 'bbox_map.json',
    'meta_file': 'stations.csv',
    'random_state': 123,
    'splits': (0.7, 0.15, 0.15),
    'batch_size': 32,
    'num_classes': 8,
    'transfer_epochs': 0,
    'finetune_epochs': 1,
    'finetune_layers': 1,
    'model_backbone': 'resnet50',
    'num_workers': 32,
    'eval_metrics': ['accuracy'],
    'finetune_callbacks': [keras.callbacks.EarlyStopping(patience=2)],
}


def main() -> None:
    """Prepare objects shared across in-sample experiments."""
    label_dict = {
        k: v for k, v in load_csv(os.path.join(CFG['root_dir'], CFG['label_file']))
    }
    stations_dict = {
        k: {'station': v}
        for k, v in load_csv(os.path.join(CFG['root_dir'], CFG['meta_file']))
    }
    mapping_dict = load_json(os.path.join(CFG['root_dir'], CFG['mapping_file']))

    augmentation = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ]
    )
    dataset = WildlifeDataset(
        keys=list(label_dict.keys()),
        image_dir=CFG['img_dir'],
        detector_file_path=os.path.join(CFG['root_dir'], CFG['detector_file']),
        label_file_path=os.path.join(CFG['root_dir'], CFG['label_file']),
        bbox_mapping_file_path=os.path.join(CFG['root_dir'], CFG['mapping_file']),
        batch_size=CFG['batch_size'],
        augmentation=augmentation,
    )

    print('---> Split data into train, val and test sets')

    keys_train, keys_val, keys_test = do_stratified_splitting(
        mapping_dict=mapping_dict,
        img_keys=list(label_dict.keys()),
        splits=CFG['splits'],
        meta_dict=stations_dict,
        random_state=CFG['random_state']
    )
    dataset_train = modify_dataset(dataset=dataset, keys=keys_train)
    dataset_val = modify_dataset(dataset=dataset, keys=keys_val)
    dataset_test = modify_dataset(dataset=dataset, keys=keys_test)

    trainer = WildlifeTrainer(
        batch_size=CFG['batch_size'],
        loss_func=keras.losses.SparseCategoricalCrossentropy(),
        num_classes=CFG['num_classes'],
        transfer_epochs=CFG['transfer_epochs'],
        finetune_epochs=CFG['finetune_epochs'],
        transfer_optimizer=keras.optimizers.Adam(),
        finetune_optimizer=keras.optimizers.Adam(),
        finetune_layers=CFG['finetune_layers'],
        model_backbone=CFG['model_backbone'],
        transfer_callbacks=None,
        finetune_callbacks=None,
        num_workers=CFG['num_workers'],
        eval_metrics=CFG['eval_metrics'],
    )

    trainer.fit(train_dataset=dataset_train, val_dataset=dataset_val)
    print('---> Fine buh-bye')


if __name__ == '__main__':
    main()