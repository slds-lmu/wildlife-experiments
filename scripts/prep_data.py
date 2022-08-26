"""Prepare data to perform experiments on."""

import os
import shutil
from typing import Dict, Final
from wildlifeml import MegaDetector
from wildlifeml.data import BBoxMapper
from wildlifeml.utils.io import load_csv_dict, save_as_csv, save_as_json

CFG: Final[Dict] = {
    'root_dir': '/common/bothmannl/',
    'img_dir': 'wildlife_images/usecase2/original_images/',
    'target_dir': '/home/wimmerl/projects/wildlife-experiments/data/',
    'info_file': 'metadata/uc2_labels.csv',
    'detector_file': 'images_megadetector.json',
    'md_conf': 0.1,
    'md_batchsize': 32,
    'label_file': 'labels.csv',
    'station_file': 'stations.csv',
}

# Copy file with meta information over

os.makedirs(CFG['target_dir'], exist_ok=True)
shutil.copyfile(
    os.path.join(CFG['root_dir'], CFG['info_file']),
    os.path.join(CFG['target_dir'], 'metadata.csv')
)

# Create label map and  label file with two columns (img key, numeric label)

info_list = load_csv_dict(os.path.join(CFG['root_dir'], CFG['info_file']))

class_names = list(set([x['true_class'] for x in info_list]))
class_names = sorted(class_names)
label_map = {class_names[i]: i for i in range(len(class_names))}
label_dict = {x['orig_name']: label_map[x['true_class']] for x in info_list}
station_dict = {x['orig_name']: x['station'] for x in info_list}

save_as_json(label_map, os.path.join(CFG['target_dir'], 'label_map.json'))
save_as_csv(
    [(k, v) for k, v in label_dict.items()],
    os.path.join(CFG['target_dir'], 'labels.csv'),
)
save_as_csv(
    [(k, v) for k, v in station_dict.items()],
    os.path.join(CFG['target_dir'], 'stations.csv'),
)

# Run MegaDetector

# md = MegaDetector(batch_size=CFG['md_batchsize'], confidence_threshold=['md_conf'])
# md.predict_directory(
#     directory=os.path.join(CFG['root_dir'], CFG['img_dir']),
#     output_file=os.path.join(CFG['target_dir'], CFG['detector_file']),
# )

# Save mapping from img to bboxes

mapping_file = os.path.join(CFG['target_dir'], 'bbox_map.json')
mapper = BBoxMapper(os.path.join(CFG['target_dir'], CFG['detector_file']))
save_as_json(mapper.get_keymap(), mapping_file)