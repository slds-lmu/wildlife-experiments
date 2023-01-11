"""Prepare objects shared across in-sample experiments."""

import click
import albumentations as A
import os
from typing import Final, List, Dict
from wildlifeml.data import WildlifeDataset, subset_dataset
from wildlifeml.utils.datasets import do_stratified_splitting, separate_empties
from wildlifeml.utils.io import (
    load_csv,
    load_json,
    save_as_pickle
)
from wildlifeml.utils.misc import flatten_list

STATIONS_IS: Final[List] = [
    '8235_For',
    '5838_2For',
    '6225_2For',
    '7935_2_F',
    '5924_3For',
    '8229_2_F',
    '6027_3For',
    '6032_4For',
    '5636_4For',
    '7545_2For',
    '6131_1For',
    '6533_4For',
    '5923_3For',
    '6137_4For',
    '5938_3For',
    '8130_2_F',
    '5837_4For',
    '6234_2For'
]
STATIONS_OOS: Final[List] = [
    '7143_4For',
    '7446_1For',
    '6035_3For',
    '8131_1_F',
    '6934_For',
    '7837_4_For',
    '8141_2For',
    '5933_4_For',
    '7544_2For',
    '8243_1For',
    '7332_For',
    '5728_2Fb',
    '6231_4For',
    '8030_1_F',
    '7730_2For',
    '5737_3_For',
    '6122_4For',
    '6034_2For',
    '5728_2Fa'
]


@click.command()
@click.option(
    '--repo_dir', '-p', help='Your personal path to this repo.', required=True
)
def main(repo_dir: str):

    cfg: Final[Dict] = load_json(os.path.join(repo_dir, 'configs/cfg.json'))

    # DATASETS -------------------------------------------------------------------------

    # Get metadata
    stations_dict = {
        k: {'station': v}
        for k, v in load_csv(os.path.join(cfg['data_dir'], 'stations.csv'))
    }
    mapping_dict = load_json(os.path.join(cfg['data_dir'], cfg['mapping_file']))

    # Define data augmentation
    augmentation = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ]
    )
    # Create base dataset with all available keys
    all_keys = list(mapping_dict.keys())
    dataset = WildlifeDataset(
        keys=all_keys,
        image_dir=cfg['img_dir'],
        detector_file_path=os.path.join(cfg['data_dir'], cfg['detector_file']),
        label_file_path=os.path.join(cfg['data_dir'], cfg['label_file']),
        bbox_map=mapping_dict,
        batch_size=cfg['batch_size'],
        augmentation=augmentation,
    )

    # Define in-sample & out-of-sample keys according to camera stations
    keys_is = [k for k in all_keys if stations_dict[k]['station'] in STATIONS_IS]
    keys_oos = [k for k in all_keys if stations_dict[k]['station'] in STATIONS_OOS]

    # Split in-sample keys into train/val/test
    keys_is_train, keys_is_val, keys_is_test = do_stratified_splitting(
        img_keys=keys_is,
        splits=cfg['splits'],
        meta_dict=stations_dict,
        random_state=cfg['random_state']
    )

    # Split out-of-sample keys into train/test (no val required in experiments)
    keys_oos_train, keys_oos_val, keys_oos_test = do_stratified_splitting(
        img_keys=keys_oos,
        splits=cfg['splits'],
        meta_dict=stations_dict,
        random_state=cfg['random_state']
    )

    # Map keys to bbxox level
    keys_is_train = flatten_list([dataset.mapping_dict[k] for k in keys_is_train])
    keys_is_val = flatten_list([dataset.mapping_dict[k] for k in keys_is_val])
    keys_is_test = flatten_list([dataset.mapping_dict[k] for k in keys_is_test])
    keys_oos_train = flatten_list([dataset.mapping_dict[k] for k in keys_oos_train])
    keys_oos_val = flatten_list([dataset.mapping_dict[k] for k in keys_oos_val])
    keys_oos_test = flatten_list([dataset.mapping_dict[k] for k in keys_oos_test])

    # Remove empty images from train/val keys (training is only based on non-empty
    # images, while evaluation on test takes all keys into account)
    _, keys_all_nonempty = separate_empties(
        detector_file_path=os.path.join(cfg['data_dir'], cfg['detector_file']),
        conf_threshold=cfg['md_conf']
    )
    keys_is_train = list(set(keys_is_train).intersection(set(keys_all_nonempty)))
    keys_is_val = list(set(keys_is_val).intersection(set(keys_all_nonempty)))
    keys_oos_train = list(set(keys_oos_train).intersection(set(keys_all_nonempty)))
    keys_oos_val = list(set(keys_oos_val).intersection(set(keys_all_nonempty)))

    # Create data subsets from different lists of keys
    for keyset, mode in zip(
            [
                keys_is_train,
                keys_is_val,
                keys_is_train + keys_is_val,
                keys_is_test,
                keys_oos_train,
                keys_oos_val,
                keys_oos_train + keys_oos_val,
                keys_oos_test,
            ],
            [
                'is_train',
                'is_val',
                'is_trainval',
                'is_test',
                'oos_train',
                'oos_val',
                'oos_trainval',
                'oos_test'
            ]
    ):
        subset = subset_dataset(dataset, keyset)
        save_as_pickle(subset, os.path.join(cfg['data_dir'], f'dataset_{mode}.pkl'))


if __name__ == '__main__':
    main()
