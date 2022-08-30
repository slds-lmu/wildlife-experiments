"""Showcase in-sample performance of proposed pipeline."""

import os
import ray
from tensorflow import keras
from typing import Dict, Final
from wildlifeml.training.trainer import WildlifeTrainer, WildlifeTuningTrainer
from wildlifeml.utils.io import load_pickle, load_json

CFG: Final[Dict] = load_json(
    '/home/wimmerl/projects/wildlife-experiments/configs/cfg_insample.json'
)
N_GPU = len(os.environ['CUDA_VISIBLE_DEVICES'])
N_CPU: Final[int] = 16

dataset_train = load_pickle(os.path.join(CFG['data_dir'], 'dataset_train.pkl'))
dataset_val = load_pickle(os.path.join(CFG['data_dir'], 'dataset_val.pkl'))
dataset_test = load_pickle(os.path.join(CFG['data_dir'], 'dataset_test.pkl'))

tuning_trainer = WildlifeTuningTrainer(
        search_space={
                'backbone': ray.tune.choice(['resnet50']),
                'transfer_learning_rate': ray.tune.choice([1e-4]),
                'finetune_learning_rate': ray.tune.choice([1e-4]),
                'batch_size': ray.tune.choice([32, 64])
        },
        loss_func=keras.losses.SparseCategoricalCrossentropy(),
        num_classes=CFG['num_classes'],
        transfer_epochs=CFG['transfer_epochs'],
        finetune_epochs=CFG['finetune_epochs'],
        finetune_layers=CFG['finetune_layers'],
        transfer_callbacks=None,
        finetune_callbacks=None,
        num_workers=CFG['num_workers'],
        eval_metrics=CFG['eval_metrics'],
        resources_per_trial={'cpu': 4, 'gpu': N_GPU},
        max_concurrent_trials=1,
        time_budget=60,
)

# trainer = WildlifeTrainer(
#         batch_size=CFG['batch_size'],
#         loss_func=keras.losses.SparseCategoricalCrossentropy(),
#         num_classes=CFG['num_classes'],
#         transfer_epochs=CFG['transfer_epochs'],
#         finetune_epochs=CFG['finetune_epochs'],
#         transfer_optimizer=keras.optimizers.Adam(),
#         finetune_optimizer=keras.optimizers.Adam(),
#         finetune_layers=CFG['finetune_layers'],
#         model_backbone=CFG['model_backbone'],
#         transfer_callbacks=None,
#         finetune_callbacks=None,
#         num_workers=CFG['num_workers'],
#         eval_metrics=CFG['eval_metrics'],
# )

tuning_trainer.fit(train_dataset=dataset_train, val_dataset=dataset_val)
print('---> Fine buh-bye')
breakpoint()