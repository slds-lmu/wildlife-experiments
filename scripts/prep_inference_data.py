"""Prepare data to perform inference on."""
import click
import os
from typing import Dict, Final
import albumentations as A
from wildlifeml.preprocessing.megadetector import MegaDetector
from wildlifeml.data import BBoxMapper
from wildlifeml.utils.io import (
    load_json,
    save_as_json,
)
from utils_ours import seed_everything

@click.command()
@click.option(
    '--repo_dir', '-p', help='Your personal path to this repo.', required=True
)
@click.option(
    '--random_seed', '-s', help='Random seed.', required=True
)

def main(repo_dir: str, random_seed: int):

    # ----------------------------------------------------------------------------------
    # GLOBAL ---------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    seed_everything(random_seed)
    cfg: Final[Dict] = load_json(os.path.join(repo_dir, 'configs/cfg.json'))

    md = MegaDetector(
        batch_size=cfg['md_batchsize'], confidence_threshold=cfg['md_conf']
    )
    md.predict_directory(
        directory=cfg['pred_img_dir'],
        output_file=os.path.join(cfg['pred_dir'], cfg['detector_file']),
    )

    # Create mapping from img to bboxes
    mapper = BBoxMapper(os.path.join(cfg['pred_dir'], cfg['detector_file']))
    key_map = mapper.get_keymap()
    save_as_json(key_map, os.path.join(cfg['pred_dir'], cfg['mapping_file']))


if __name__ == '__main__':
    main()
