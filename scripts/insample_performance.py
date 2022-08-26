"""Showcase in-sample performance of proposed pipeline."""

from tensorflow import keras

from wildlifeml.training.trainer import WildlifeTrainer
from wildlifeml.training.evaluator import Evaluator

import scripts.prep_insample as prep
CFG = prep.CFG
dataset_train = prep.dataset_train
dataset_val = prep.dataset_val


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