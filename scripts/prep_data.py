"""Prepare data to perform experiments on."""
import collections
import math

import click
import os
from typing import Dict, Final, List
import albumentations as A
from wildlifeml.preprocessing.megadetector import MegaDetector
from wildlifeml.data import BBoxMapper, WildlifeDataset, subset_dataset, DatasetConverter
from wildlifeml.utils.datasets import do_stratified_splitting
from wildlifeml.utils.io import (
    load_csv_dict,
    save_as_csv,
    load_json,
    save_as_json,
    save_as_pickle
)
from wildlifeml.utils.misc import flatten_list
from utils_ours import seed_everything
import json
import random
import pandas as pd


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
@click.option(
    '--random_seed', '-s', help='Random seed.', required=True
)
@click.option(
    '--dataset_converter', '-c', help='Whether the data is in necessary format. ', required=False, default=False
)
@click.option(
    '--images_dir', '-i', help='Path to images if convertion needed.', required=False, default=""
)

def main(repo_dir: str, random_seed: int, dataset_converter: bool, images_dir: str):

    # ----------------------------------------------------------------------------------
    # GLOBAL ---------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    seed_everything(random_seed)
    cfg: Final[Dict] = load_json(os.path.join(repo_dir, 'configs/cfg.json'))

    if dataset_converter:
        print("Converting the data to the required format...")
        conv = DatasetConverter(root_dir=images_dir, target_dir=cfg['img_dir'])
        conv.convert()

    # check if metadata.csv exists 
    if 'metadata.csv' not in os.listdir(cfg['data_dir']):
        # prepare the metadata file
        stations = STATIONS_IS + STATIONS_OOS
        with open(os.path.join(cfg['data_dir'], "label_map.json")) as json_file:
            label_map = json.load(json_file)
        img_names = os.listdir(cfg['img_dir'])
        true_class = []
        filtered_names = []
        meta_stations = []
        doy = []
        hour = []
        station_sets = ["s1", "s2"]
        station_set = []
        for filename in img_names:
            species_name = filename.split('_')[:2]
            species_name = "_".join(species_name)
            if species_name in label_map:
                true_class.append(species_name)
                filtered_names.append(filename)
                meta_stations.append(random.choice(stations))
                station_set.append(random.choice(station_sets))
                doy.append(random.randint(1,365))
                hour.append(random.randint(0,23))
        metadata = pd.DataFrame()
        metadata['orig_name'] = filtered_names
        metadata['true_class'] = true_class
        metadata['doy'] = doy
        metadata['hour'] = hour
        metadata['station'] = meta_stations
        metadata['station_set'] = station_set
        metadata.to_csv(os.path.join(cfg['data_dir'], "metadata.csv"))

        
    # Create label map and  label file with two columns (img key, numeric label)
    info_list = load_csv_dict(os.path.join(cfg['data_dir'], cfg['info_file']))
    class_names = list(set([x['true_class'] for x in info_list]))
    class_names = sorted(class_names)
    label_map = {class_names[i]: i for i in range(len(class_names))}
    label_dict = {x['orig_name']: label_map[x['true_class']] for x in info_list}
    station_dict = {x['orig_name']: x['station'] for x in info_list}


    # Run MegaDetector
    if not os.path.exists(os.path.join(cfg['data_dir'], cfg['detector_file'])):
        md = MegaDetector(
            batch_size=cfg['md_batchsize'], confidence_threshold=cfg['md_conf']
        )
        md.predict_directory(
            directory=cfg['img_dir'],
            output_file=os.path.join(cfg['data_dir'], cfg['detector_file']),
        )

    # Create mapping from img to bboxes
    mapper = BBoxMapper(os.path.join(cfg['data_dir'], cfg['detector_file']))
    key_map = mapper.get_keymap()

    # Eliminate imgs with missing information
    for k in (set(key_map) - set(label_dict)):
        del key_map[k]
    for k in (set(key_map) - set(station_dict)):
        del key_map[k]
    # Rm class 'other'
    other = list(set(k for k, v in label_dict.items() if v == 6))
    for k in other:
        if k in key_map.keys():
            del key_map[k]

    # Save everything
    save_as_json(label_map, os.path.join(cfg['data_dir'], 'label_map.json'))
    save_as_csv(
        [(k, v) for k, v in label_dict.items()],
        os.path.join(cfg['data_dir'], cfg['label_file']),
    )
    save_as_csv(
        [(k, v) for k, v in station_dict.items()],
        os.path.join(cfg['data_dir'], cfg['meta_file']),
    )
    save_as_json(key_map, os.path.join(cfg['data_dir'], cfg['mapping_file']))

    # ----------------------------------------------------------------------------------
    # DATA SPLITS ----------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    # Define data augmentation
    augmentation = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ]
    )
    # Create base dataset with all available keys
    all_keys = list(key_map.keys())
    dataset = WildlifeDataset(
        keys=all_keys,
        image_dir=cfg['img_dir'],
        detector_file_path=os.path.join(cfg['data_dir'], cfg['detector_file']),
        label_file_path=os.path.join(cfg['data_dir'], cfg['label_file']),
        bbox_map=key_map,
        batch_size=cfg['batch_size'],
        augmentation=augmentation,
    )

    # Restructure station dict to match stratified-splitting arguments
    station_dict_mod = {k: {'station': v} for k, v in station_dict.items()}
    # Define in-sample & out-of-sample keys according to camera stations
    if os.path.exists(os.path.join(cfg['data_dir'], 'stations_split.json')):
        stations = load_json(os.path.join(cfg['data_dir'], 'stations_split.json'))
        stations_is = stations['STATIONS_IS']
        stations_oos = stations['STATIONS_OOS']
    else:
        random.seed(random_seed)
        stations = list(set(station_dict.values()))
        stations_is = list(random.sample(stations, math.ceil(0.5 * len(stations))))
        stations_oos = list(set(stations) - set(stations_is))
        stations_dct = {'STATIONS_IS': stations_is, 'STATIONS_OOS': stations_oos}
        save_as_json(stations_dct, os.path.join(cfg['data_dir'], 'stations_split.json')) 

    keys_is = [k for k in all_keys if station_dict_mod[k]['station'] in stations_is]
    keys_oos = [k for k in all_keys if station_dict_mod[k]['station'] in stations_oos]

    labels_keys_is = [label_dict[k] for k in keys_is]
    labels_keys_oos = [label_dict[k] for k in keys_oos]
    counter_is = collections.Counter(labels_keys_is)
    counter_oos = collections.Counter(labels_keys_oos)

    # Split keys into train/val/test (only two-way for in-sample bc splitting is done
    # later according to chosen MD threshold)

    keys_is_train, keys_is_val, keys_is_test = do_stratified_splitting(
        img_keys=keys_is,
        splits=cfg['splits'],
        meta_dict=station_dict_mod,
        random_state=random_seed,
    )

    keys_oos_train, keys_oos_val, keys_oos_test = do_stratified_splitting(
        img_keys=keys_oos,
        splits=cfg['splits'],
        meta_dict=station_dict_mod,
        random_state=random_seed,
    )
    keys_train, _, keys_val = do_stratified_splitting(
        img_keys=all_keys,
        splits=(0.7/0.85, 0., 0.15/0.85),
        meta_dict=station_dict_mod,
        random_state=random_seed,
    )

    # Map keys to bbxox level
    keys_is_train_bb = flatten_list([dataset.mapping_dict[k] for k in keys_is_train])
    keys_is_val_bb = flatten_list([dataset.mapping_dict[k] for k in keys_is_val])
    keys_is_test_bb = flatten_list([dataset.mapping_dict[k] for k in keys_is_test])
    keys_oos_train_bb = flatten_list([dataset.mapping_dict[k] for k in keys_oos_train])
    keys_oos_val_bb = flatten_list([dataset.mapping_dict[k] for k in keys_oos_val])
    keys_oos_test_bb = flatten_list([dataset.mapping_dict[k] for k in keys_oos_test])
    keys_train_bb = flatten_list([dataset.mapping_dict[k] for k in keys_train])
    keys_val_bb = flatten_list([dataset.mapping_dict[k] for k in keys_val])

    # Create data subsets from different lists of keys
    for keyset, mode in zip(
            [
                keys_is_train_bb,
                keys_is_val_bb,
                keys_is_train_bb + keys_is_val_bb,
                keys_is_test_bb,
                keys_oos_train_bb,
                keys_oos_val_bb,
                keys_oos_train_bb + keys_oos_val_bb,
                keys_oos_test_bb,
                keys_train_bb,
                keys_val_bb,
            ],
            [
                'is_train',
                'is_val',
                'is_train_val',
                'is_test',
                'oos_train',
                'oos_val',
                'oos_trainval',
                'oos_test',
                'full_train',
                'full_val',
            ]
    ):
        subset = subset_dataset(dataset, keyset)
        save_as_pickle(subset, os.path.join(cfg['data_dir'], f'dataset_{mode}.pkl'))


if __name__ == '__main__':
    main()
