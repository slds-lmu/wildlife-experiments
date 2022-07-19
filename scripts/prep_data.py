"""Prepare data to perform experiments on."""

import os
import shutil
from typing import Final
from wildlifeml.utils.io import load_csv_dict, save_as_csv, save_as_json

ROOT_DIR: Final[str] = '/common/bothmannl/'
TARGET_DIR: Final[str] = '/home/wimmerl/projects/wildlife-experiments/data/'
INFO_FILE: Final[str] = 'metadata/uc2_labels.csv'

# Copy file with meta information over

os.makedirs(TARGET_DIR, exist_ok=True)
shutil.copyfile(
    os.path.join(ROOT_DIR, INFO_FILE), os.path.join(TARGET_DIR, 'metadata.csv')
)

# Create label map and  label file with two columns (img key, numeric label)

info_dict = {k: v for k, v in load_csv_dict(os.path.join(ROOT_DIR, INFO_FILE))}
class_names = list(set([v['true_class'] for v in info_dict.values()]))
class_names = sorted(class_names)

label_map = {class_names[i]: i for i in range(len(class_names))}
label_dict = {k: label_map[v['true_class']] for k, v in info_dict}
save_as_json(label_map, os.path.join(ROOT_DIR, 'label_map.json'))
save_as_csv(
    [(k, v) for k, v in label_dict.items()],
    os.path.join(ROOT_DIR, 'labels.csv'),
)
