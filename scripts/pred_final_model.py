"""Use final model to predict on new data."""
import os
from typing import Final, Dict, List
import albumentations as A

from tensorflow.keras.models import load_model
from wildlifeml import WildlifeDataset
from wildlifeml.data import BBoxMapper
from wildlifeml.training.evaluator import Evaluator
from wildlifeml.utils.io import load_json

IMG_DIR: Final[str] = '/common/bothmannl/wildlife_images/usecase2/original_images/'
PROJ_DIR: Final[str] = '/home/wimmerl/projects/wildlife-experiments/'
PRED_DIR: Final[str] = PROJ_DIR + 'results/bavaria/md5/predictions/'
DETECTOR_FILE_PATH: Final[str] = PROJ_DIR + 'data/bavaria/md5/md.json'
BBOX_PATH: Final[str] = PROJ_DIR + 'data/bavaria/md5/bbox_map.json'
MODEL_PATH: Final[str] = PROJ_DIR + 'data/bavaria/md5/final_ckpt/ckpt_final.hdf5'
THRESHOLD: Final[float] = 0.1
KEYS: Final[List] = ['10000_I_00249b.JPG', '10001_I_00058a.JPG', '10003_I_00226c.JPG']
BATCH_SIZE: Final[int] = 64
NUM_CLASSES: Final[int] = 7
EMPTY_CLASS_ID: Final[int] = 0


def main():

    mapper = BBoxMapper(DETECTOR_FILE_PATH)
    key_map = mapper.get_keymap()
    all_keys = list(key_map.keys())
    dataset = WildlifeDataset(
        keys=[k for k in all_keys if any(x in k for x in KEYS)],
        image_dir=IMG_DIR,
        detector_file_path=DETECTOR_FILE_PATH,
        bbox_map=load_json(BBOX_PATH),
        batch_size=BATCH_SIZE,
        augmentation=A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ]
        ),
    )

    model = load_model(MODEL_PATH)
    evaluator = Evaluator(
        detector_file_path=DETECTOR_FILE_PATH,
        num_classes=NUM_CLASSES,
        empty_class_id=EMPTY_CLASS_ID,
        conf_threshold=THRESHOLD,
        dataset=dataset
    )
    evaluator.evaluate(model)
    os.makedirs(PRED_DIR, exist_ok=True)
    evaluator.save_predictions(file_path=PRED_DIR + 'preds_img.csv', img_level=True)
    evaluator.save_predictions(file_path=PRED_DIR + 'preds_bbox.csv', img_level=False)


if __name__ == '__main__':
    main()
