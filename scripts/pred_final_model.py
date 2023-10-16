"""Use final model to predict on new data."""
import os
from typing import Final, Dict, List
import albumentations as A
import click

from tensorflow.keras.models import load_model
from wildlifeml import WildlifeDataset
from wildlifeml.data import BBoxMapper
from wildlifeml.training.evaluator import Evaluator
from wildlifeml.utils.io import load_json
from wildlifeml.utils.misc import flatten_list
from keras.applications import imagenet_utils

THRESHOLD: Final[float] = 0.1
EMPTY_CLASS_ID: Final[int] = 0

@click.command()
@click.option(
    '--repo_dir', '-p', help='Your personal path to this repo.', required=True
)
@click.option(
    '--model_path', '-m', help='Your personal path to the model you want to use for predictions.', required=True
)
@click.option(
    '--output_path', '-o', help='Your personal path to where you want to save the predictions.', required=True
)

def main(repo_dir: str, model_path: str, output_path: str):

    cfg: Final[Dict] = load_json(os.path.join(repo_dir, 'configs/cfg.json'))
    mapper = BBoxMapper(os.path.join(cfg['pred_dir'], cfg['detector_file']))
    key_map = mapper.get_keymap()
    keys = os.listdir(cfg['pred_img_dir'])

    dataset = WildlifeDataset(
        keys=flatten_list([key_map[k] for k in keys]),
        image_dir=cfg['pred_img_dir'],
        detector_file_path=os.path.join(cfg['pred_dir'], cfg['detector_file']),
        bbox_map=load_json(os.path.join(cfg['pred_dir'], cfg['mapping_file'])),
        batch_size=cfg['batch_size'],
        augmentation=A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ]
        ),
    )

    model = load_model(model_path, custom_objects={'imagenet_utils': imagenet_utils})
    evaluator = Evaluator(
        detector_file_path=os.path.join(cfg['pred_dir'], cfg['detector_file']),
        num_classes=cfg['num_classes'],
        empty_class_id=EMPTY_CLASS_ID,
        conf_threshold=THRESHOLD,
        dataset=dataset
    )
    evaluator.evaluate(model)
    os.makedirs(output_path, exist_ok=True)
    evaluator.save_predictions(filepath=os.path.join(output_path, 'preds_img.csv'), img_level=True)
    evaluator.save_predictions(filepath=os.path.join(output_path, 'preds_bbox.csv'), img_level=False)


if __name__ == '__main__':
    main()
