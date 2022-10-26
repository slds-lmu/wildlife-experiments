"""Prepare data to perform experiments on."""

import click
import os
from typing import Dict, Final
from wildlifeml.preprocessing.megadetector import MegaDetector
from wildlifeml.data import BBoxMapper
from wildlifeml.utils.io import load_csv_dict, save_as_csv, load_json, save_as_json


@click.command()
@click.option(
    '--repo_dir', '-p', help='Your personal path to this repo.', required=True
)
def main(repo_dir: str):

    cfg: Final[Dict] = load_json(os.path.join(repo_dir, 'configs/cfg.json'))

    # Create label map and  label file with two columns (img key, numeric label)
    info_list = load_csv_dict(os.path.join(cfg['data_dir'], 'metadata.csv'))
    class_names = list(set([x['true_class'] for x in info_list]))
    class_names = sorted(class_names)
    label_map = {class_names[i]: i for i in range(len(class_names))}
    label_dict = {x['orig_name']: label_map[x['true_class']] for x in info_list}
    station_dict = {x['orig_name']: x['station'] for x in info_list}

    # Run MegaDetector
    if not os.path.exists(os.path.join(cfg['data_dir'], cfg['detector_file'])):
        md = MegaDetector(
            batch_size=cfg['md_batchsize'], confidence_threshold=['md_conf']
        )
        md.predict_directory(
            directory=os.path.join(cfg['root_dir'], cfg['img_dir']),
            output_file=os.path.join(cfg['data_dir'], cfg['detector_file']),
        )

    # Create mapping from img to bboxes
    mapping_file = os.path.join(cfg['data_dir'], 'bbox_map.json')
    mapper = BBoxMapper(os.path.join(cfg['data_dir'], cfg['detector_file']))
    key_map = mapper.get_keymap()

    # Eliminate imgs with missing information
    for k in (set(key_map) - set(label_dict)):
        del key_map[k]
    for k in (set(key_map) - set(station_dict)):
        del key_map[k]

    # Save everything
    save_as_json(label_map, os.path.join(cfg['data_dir'], 'label_map.json'))
    save_as_csv(
        [(k, v) for k, v in label_dict.items()],
        os.path.join(cfg['data_dir'], 'labels.csv'),
    )
    save_as_csv(
        [(k, v) for k, v in station_dict.items()],
        os.path.join(cfg['data_dir'], 'stations.csv'),
    )
    save_as_json(key_map, mapping_file)


if __name__ == '__main__':
    main()
