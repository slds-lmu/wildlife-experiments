"""Prepare objects shared across in-sample experiments."""

import albumentations as A
import os
from typing import Final, Dict

from wildlifeml.data import WildlifeDataset, subset_dataset
from wildlifeml.utils.datasets import (
    do_stratified_splitting,
    do_stratified_cv,
    map_bbox_to_img,
)
from wildlifeml.utils.io import load_csv, load_csv_dict, load_json, save_as_pickle
from wildlifeml.utils.misc import flatten_list

CFG: Final[Dict] = load_json(
    '/home/wimmerl/projects/wildlife-experiments/configs/cfg_insample.json'
)

label_dict = {
    k: v for k, v in load_csv(os.path.join(CFG['data_dir'], CFG['label_file']))
}
stations_dict = {
    k: {'station': v}
    for k, v in load_csv(os.path.join(CFG['data_dir'], CFG['meta_file']))
}
detector_dict = load_json(os.path.join(CFG['data_dir'], CFG['detector_file']))
mapping_dict = load_json(os.path.join(CFG['data_dir'], CFG['mapping_file']))

augmentation = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Rotate(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ]
)
dataset = WildlifeDataset(
    keys=flatten_list([v for v in mapping_dict.values()]),
    image_dir=CFG['img_dir'],
    detector_file_path=os.path.join(CFG['data_dir'], CFG['detector_file']),
    label_file_path=os.path.join(CFG['data_dir'], CFG['label_file']),
    bbox_map=mapping_dict,
    batch_size=CFG['batch_size'],
    augmentation=augmentation,
)

print('---> Split data into train, val and test sets')
keys_train, keys_val, keys_test = do_stratified_splitting(
    img_keys=list(set([map_bbox_to_img(k) for k in dataset.keys])),
    splits=CFG['splits'],
    meta_dict=stations_dict,
    random_state=CFG['random_state']
)
for keyset, mode in zip([keys_train, keys_val, keys_test], ['train', 'val', 'test']):
    subset = subset_dataset(
        dataset, flatten_list([dataset.mapping_dict[k] for k in keyset])
    )
    save_as_pickle(subset, os.path.join(CFG['data_dir'], f'dataset_{mode}.pkl'))