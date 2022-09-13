"""Prepare objects shared across in-sample experiments."""

import albumentations as A
import os
from typing import Final, Dict, List

from wildlifeml.data import WildlifeDataset, subset_dataset
from wildlifeml.utils.datasets import (
    do_stratified_splitting,
    do_stratified_cv,
    map_bbox_to_img,
    separate_empties,
)
from wildlifeml.utils.io import (
    load_csv, load_csv_dict, load_json, load_pickle, save_as_pickle
)
from wildlifeml.utils.misc import flatten_list

CFG: Final[Dict] = load_json(
    '/home/wimmerl/projects/wildlife-experiments/configs/cfg_insample.json'
)
STATIONS_IS: Final[List] = [
    '8235_For',
    '5838_2For',
    '6225_2For',
    # '7935_2_F',
    # '5924_3For',
    # '8229_2_F',
    # '6027_3For',
    # '6032_4For',
    # '5636_4For',
    # '7545_2For',
    # '6131_1For',
    # '6533_4For',
    # '5923_3For',
    # '6137_4For',
    # '5938_3For',
    # '8130_2_F',
    # '5837_4For',
    # '6234_2For'
]
STATIONS_OOS: Final[List] = [
    '5923_3For',
    '6137_4For',
    '5938_3For',
    # '8130_2_F',
    # '5837_4For',
    # '6234_2For',
    # '7143_4For',
    # '7446_1For',
    # '6035_3For',
    # '8131_1_F',
    # '6934_For',
    # '7837_4_For',
    # '8141_2For',
    # '5933_4_For',
    # '7544_2For',
    # '8243_1For',
    # '7332_For',
    # '5728_2Fb',
    # '6231_4For',
    # '8030_1_F',
    # '7730_2For',
    # '5737_3_For',
    # '6122_4For',
    # '6034_2For',
    # '5728_2Fa'
]

# DATASETS -----------------------------------------------------------------------------

# Get metadata

label_dict = {
    k: v for k, v in load_csv(os.path.join(CFG['data_dir'], CFG['label_file']))
}
stations_dict = {
    k: {'station': v}
    for k, v in load_csv(os.path.join(CFG['data_dir'], CFG['meta_file']))
}
detector_dict = load_json(os.path.join(CFG['data_dir'], CFG['detector_file']))
mapping_dict = load_json(os.path.join(CFG['data_dir'], CFG['mapping_file']))

# Prepare dataset and perform train-val-test split

augmentation = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Rotate(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ]
)

_, nonempty_keys_md = separate_empties(
    detector_file_path=os.path.join(CFG['data_dir'], CFG['detector_file']),
    conf_threshold=CFG['md_conf']
)
nonempty_keys = list(
    set(nonempty_keys_md).intersection(
        set(flatten_list([v for v in mapping_dict.values()]))
    )
)
dataset = WildlifeDataset(
    keys=nonempty_keys,
    image_dir=CFG['img_dir'],
    detector_file_path=os.path.join(CFG['data_dir'], CFG['detector_file']),
    label_file_path=os.path.join(CFG['data_dir'], CFG['label_file']),
    bbox_map=mapping_dict,
    batch_size=CFG['batch_size'],
    augmentation=augmentation,
)

# print('---> Split data into train, val and test sets')
# keys_train, keys_val, keys_test = do_stratified_splitting(
#     img_keys=list(set([map_bbox_to_img(k) for k in dataset.keys])),
#     splits=CFG['splits'],
#     meta_dict=stations_dict,
#     random_state=CFG['random_state']
# )
# for keyset, mode in zip(
#         [keys_train, keys_val, keys_train + keys_val, keys_test],
#         ['train', 'val', 'trainval', 'test']):
#     subset = subset_dataset(
#         dataset, flatten_list([dataset.mapping_dict[k] for k in keyset])
#     )
#     save_as_pickle(subset, os.path.join(CFG['data_dir'], f'dataset_{mode}.pkl'))

# DATASETS OUT-OF-SAMPLE ---------------------------------------------------------------

keys_is = [
    map_bbox_to_img(k) for k in nonempty_keys
    if stations_dict[map_bbox_to_img(k)]['station'] in STATIONS_IS
]
keys_oos = [
    map_bbox_to_img(k) for k in nonempty_keys
    if stations_dict[map_bbox_to_img(k)]['station'] in STATIONS_OOS
]

keys_is_train, keys_is_val, keys_is_test = do_stratified_splitting(
    img_keys=keys_is,
    splits=CFG['splits'],
    meta_dict=stations_dict,
    random_state=CFG['random_state']
)
keys_oos_train, _, keys_oos_test = do_stratified_splitting(
    img_keys=keys_oos,
    splits=(CFG['splits'][0] + CFG['splits'][1], 0.0, CFG['splits'][2]),
    meta_dict=stations_dict,
    random_state=CFG['random_state']
)

for keyset, mode in zip(
        [
            keys_is_train,
            keys_is_val,
            keys_is_train + keys_is_val,
            keys_is_test,
            keys_oos_train,
            keys_oos_test,
        ],
        [
            'is_train',
            'is_val',
            'is_trainval',
            'is_test',
            'oos_train',
            'oos_test'
        ]
):
    subset = subset_dataset(
        dataset, flatten_list([dataset.mapping_dict[k] for k in keyset])
    )
    save_as_pickle(subset, os.path.join(CFG['data_dir'], f'dataset_{mode}.pkl'))
