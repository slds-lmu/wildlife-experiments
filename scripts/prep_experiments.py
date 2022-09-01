"""Prepare objects shared across in-sample experiments."""

import albumentations as A
import os
from tensorflow import keras
from typing import Final, Dict

from wildlifeml.data import WildlifeDataset, subset_dataset
from wildlifeml.training.trainer import WildlifeTrainer, WildlifeTuningTrainer
from wildlifeml.training.evaluator import Evaluator
from wildlifeml.training.active import ActiveLearner
from wildlifeml.utils.datasets import (
    do_stratified_splitting,
    do_stratified_cv,
    map_bbox_to_img,
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

# DATASETS IN-SAMPLE -------------------------------------------------------------------

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

dataset_pool = subset_dataset(
    dataset, flatten_list([dataset.mapping_dict[k] for k in keys_train + keys_val])
)
save_as_pickle(dataset_pool, os.path.join(CFG['data_dir'], 'dataset_pool.pkl'))

# DATASETS OUT-OF-SAMPLE ---------------------------------------------------------------

keys_is = [k for k, v in stations_dict.keys() if v in STATIONS_IS]
keys_is_train, _, keys_is_val = do_stratified_splitting(
    img_keys=list(set([map_bbox_to_img(k) for k in dataset.keys])),
    splits=(CFG['splits'][0] + CFG['splits'][2], 0., CFG['splits'[1]]),
    meta_dict=stations_dict,
    random_state=CFG['random_state']
)
keys_oos = [k for k, v in stations_dict.keys() if v in STATIONS_OOS]

for keyset, mode in zip(
        [keys_is_train, keys_is_val, keys_oos], ['is_train', 'is_val, ''oos']
):
    subset = subset_dataset(
        dataset, flatten_list([dataset.mapping_dict[k] for k in keyset])
    )
    save_as_pickle(subset, os.path.join(CFG['data_dir'], f'dataset_{mode}.pkl'))

# TRAINERS -----------------------------------------------------------------------------

N_GPU = len(os.environ['CUDA_VISIBLE_DEVICES'])
N_CPU: Final[int] = 16

# Set up wildlife trainers and evaluator

# tuning_trainer = WildlifeTuningTrainer(
#     search_space={
#         'backbone': ray.tune.choice(['resnet50']),
#         'transfer_learning_rate': ray.tune.choice([1e-4]),
#         'finetune_learning_rate': ray.tune.choice([1e-4]),
#         'batch_size': ray.tune.choice([32, 64])
#     },
#     loss_func=keras.losses.SparseCategoricalCrossentropy(),
#     num_classes=CFG['num_classes'],
#     transfer_epochs=CFG['transfer_epochs'],
#     finetune_epochs=CFG['finetune_epochs'],
#     finetune_layers=CFG['finetune_layers'],
#     transfer_callbacks=None,
#     finetune_callbacks=None,
#     num_workers=CFG['num_workers'],
#     eval_metrics=CFG['eval_metrics'],
#     resources_per_trial={'cpu': 4, 'gpu': N_GPU},
#     max_concurrent_trials=1,
#     time_budget=60,
# )

trainer = WildlifeTrainer(
    batch_size=CFG['batch_size'],
    loss_func=keras.losses.SparseCategoricalCrossentropy(),
    num_classes=CFG['num_classes'],
    transfer_epochs=CFG['transfer_epochs'],
    finetune_epochs=CFG['finetune_epochs'],
    transfer_optimizer=keras.optimizers.Adam(),
    finetune_optimizer=keras.optimizers.Adam(),
    finetune_layers=CFG['finetune_layers'],
    model_backbone=CFG['model_backbone'],
    transfer_callbacks=None,
    finetune_callbacks=None,
    num_workers=CFG['num_workers'],
    eval_metrics=CFG['eval_metrics'],
)

dataset_test = load_pickle(os.path.join(CFG['data_dir'], 'dataset_test.pkl'))
evaluator = Evaluator(
    label_file_path=os.path.join(CFG['data_dir'], CFG['label_file']),
    detector_file_path=os.path.join(CFG['data_dir'], CFG['detector_file']),
    dataset=dataset_test,
    num_classes=trainer.get_num_classes(),

)

active_learner = ActiveLearner(
    trainer=trainer,
    pool_dataset=dataset_pool,
    label_file_path=os.path.join(CFG['data_dir'], CFG['label_file']),
    empty_class_id=load_json(os.path.join(
        CFG['data_dir'], 'label_map.json')).get('empty'),
    acquisitor_name='entropy',
    train_size=CFG['splits'][0],
    test_dataset=dataset_test,
    test_logfile_path=os.path.join(CFG['data_dir'], CFG['test_logfile.json']),
    meta_dict=stations_dict,
)

# save_as_pickle(tuning_trainer, os.path.join(CFG['data_dir'], 'tuning_trainer.pkl'))
save_as_pickle(trainer, os.path.join(CFG['data_dir'], 'trainer.pkl'))
save_as_pickle(evaluator, os.path.join(CFG['data_dir'], 'evaluator.pkl'))
save_as_pickle(active_learner, os.path.join(CFG['data_dir'], 'active_learner.pkl'))