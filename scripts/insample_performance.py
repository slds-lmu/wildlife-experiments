"""Showcase in-sample performance of proposed pipeline."""

import os
from tensorflow import keras
from typing import Dict, Final
from wildlifeml.training.trainer import WildlifeTrainer, WildlifeTuningTrainer
from wildlifeml.utils.io import load_pickle, load_json

print(os.cpu_count())
break

CFG: Final[Dict] = load_json(
    '/home/wimmerl/projects/wildlife-experiments/configs/cfg_insample.json'
)
dataset_train = load_pickle(os.path.join(CFG['data_dir'], 'dataset_train.pkl'))
dataset_val = load_pickle(os.path.join(CFG['data_dir'], 'dataset_val.pkl'))
dataset_test = load_pickle(os.path.join(CFG['data_dir'], 'dataset_test.pkl'))

trainer_tuned = WildlifeTuningTrainer(
        loss_func=keras.losses.SparseCategoricalCrossentropy(),
        num_classes=CFG['num_classes'],
        transfer_epochs=CFG['transfer_epochs'],
        finetune_epochs=CFG['finetune_epochs'],
        transfer_optimizer=keras.optimizers.Adam(),
        finetune_optimizer=keras.optimizers.Adam(),
        finetune_layers=CFG['finetune_layers'],
        transfer_callbacks=None,
        finetune_callbacks=None,
        num_workers=CFG['num_workers'],
        eval_metrics=CFG['eval_metrics'],
)
"""
search_space: Dict,
        loss_func: Any,
        num_classes: int,
        transfer_epochs: int,
        finetune_epochs: int,
        finetune_layers: int,
        num_workers: int = 0,
        transfer_callbacks: Optional[List] = None,
        finetune_callbacks: Optional[List] = None,
        eval_metrics: Optional[List[str]] = None,
        local_dir: str = './ray_results/',
        random_state: int = 123,
        resources_per_trial: Optional[Dict] = None,
        max_concurrent_trials: int = 2,
        objective: str = 'val_accuracy',
        mode: str = 'max',
        n_trials: int = 2,
        transfer_epochs_per_trial: Optional[int] = 1,
        finetune_epochs_per_trial: Optional[int] = 1,
        time_budget: int = 3600,
        verbose: int = 0,
        search_alg_id: str = 'hyperoptsearch',
        scheduler_alg_id: str = 'ashascheduler',
"""

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

trainer.fit(train_dataset=dataset_train, val_dataset=dataset_val)
print('---> Fine buh-bye')
breakpoint()