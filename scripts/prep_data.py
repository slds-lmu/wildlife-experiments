"""Prepare data to perform experiments on."""

import os
import shutil
from typing import Dict, Final
from wildlifeml import MegaDetector
from wildlifeml.data import BBoxMapper
from wildlifeml.utils.io import load_csv_dict, save_as_csv, load_json, save_as_json

CFG: Final[Dict] = load_json(
    '/home/wimmerl/projects/wildlife-experiments/configs/cfg.json'
)

# Copy file with meta information over
os.makedirs(CFG['data_dir'], exist_ok=True)
shutil.copyfile(
    os.path.join(CFG['root_dir'], CFG['info_file']),
    os.path.join(CFG['data_dir'], 'metadata.csv')
)

# Create label map and  label file with two columns (img key, numeric label)
info_list = load_csv_dict(os.path.join(CFG['root_dir'], CFG['info_file']))
class_names = list(set([x['true_class'] for x in info_list]))
class_names = sorted(class_names)
label_map = {class_names[i]: i for i in range(len(class_names))}
label_dict = {x['orig_name']: label_map[x['true_class']] for x in info_list}
station_dict = {x['orig_name']: x['station'] for x in info_list}

# Run MegaDetector
if not os.path.exists(os.path.join(CFG['data_dir'], CFG['detector_file'])):
    md = MegaDetector(batch_size=CFG['md_batchsize'], confidence_threshold=['md_conf'])
    md.predict_directory(
        directory=os.path.join(CFG['root_dir'], CFG['img_dir']),
        output_file=os.path.join(CFG['data_dir'], CFG['detector_file']),
    )
detector_dict = load_json(os.path.join(CFG['data_dir'], CFG['detector_file']))

# Create mapping from img to bboxes
mapping_file = os.path.join(CFG['data_dir'], 'bbox_map.json')
mapper = BBoxMapper(os.path.join(CFG['data_dir'], CFG['detector_file']))
key_map = mapper.get_keymap()

# Eliminate imgs with missing information
for k in (set(key_map) - set(label_dict)):
    del key_map[k]
for k in (set(key_map) - set(station_dict)):
    del key_map[k]

# Save everything
save_as_json(label_map, os.path.join(CFG['data_dir'], 'label_map.json'))
save_as_csv(
    [(k, v) for k, v in label_dict.items()],
    os.path.join(CFG['data_dir'], 'labels.csv'),
)
save_as_csv(
    [(k, v) for k, v in station_dict.items()],
    os.path.join(CFG['data_dir'], 'stations.csv'),
)
save_as_json(key_map, mapping_file)