"""In-sample results."""

from copy import deepcopy
import numpy as np
import os
import ray
from tensorflow import keras
from typing import Dict, Final
from wildlifeml.training.trainer import WildlifeTrainer, WildlifeTuningTrainer
from wildlifeml.training.active import ActiveLearner
from wildlifeml.training.evaluator import Evaluator
from wildlifeml.utils.datasets import separate_empties, map_preds_to_img
from wildlifeml.utils.io import load_csv, load_json, load_pickle

CFG: Final[Dict] = load_json(
    '/home/wimmerl/projects/wildlife-experiments/configs/cfg_insample.json'
)
N_GPU = len(os.environ['CUDA_VISIBLE_DEVICES'])
N_CPU: Final[int] = 16

# EMPTY VS NON-EMPTY -------------------------------------------------------------------

THRESH_PROGRESSIVE: Final[float] = 0.5
THRESH_NOROUZZADEH: Final[float] = 0.9

# Get metadata

detector_dict = load_json(os.path.join(CFG['data_dir'], CFG['detector_file']))
label_dict = {
    k: v for k, v in load_csv(os.path.join(CFG['data_dir'], CFG['label_file']))
}
stations_dict = {
    k: v for k, v in load_csv(os.path.join(CFG['data_dir'], 'stations.csv'))
}

# Get truly empty images

empty_class = load_json(os.path.join(CFG['data_dir'], 'label_map.json')).get('empty')
true_empty = set([k for k, v in label_dict.items() if v == empty_class])
true_nonempty = set(label_dict.keys()) - set(true_empty)

# Compute empty-detection performance of MD

results_empty_md = {
    'names': ['ours', 'progressive', 'norouzzadeh'],
    'thresholds': [CFG['md_conf'], THRESH_PROGRESSIVE, THRESH_NOROUZZADEH]
}
tnr, tpr, fnr, fpr = [], [], [], []

for threshold in results_empty_md['thresholds']:
    keys_empty, keys_nonempty = separate_empties(
        os.path.join(CFG['data_dir'], CFG['detector_file']), threshold
    )
    tn = len(true_empty.intersection(set(keys_empty)))
    tp = len(true_nonempty.intersection(set(keys_nonempty)))
    fn = len(true_nonempty.intersection(set(keys_empty)))
    fp = len(true_empty.intersection(set(keys_nonempty)))
    tnr.append(tn / (tn + fp) if (tn + fp) > 0 else 0.)
    tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0.)
    fnr.append(fn / (tp + fn) if (tp + fn) > 0 else 0.)
    fpr.append(fp / (tn + fp) if (tn + fp) > 0 else 0.)

results_empty_md.update({'tnr': tnr, 'tpr': tpr, 'fnr': fnr, 'fpr': fpr})

# Compute empty-detection performance of pipeline

dataset_train = load_pickle(os.path.join(CFG['data_dir'], 'dataset_train.pkl'))
dataset_val = load_pickle(os.path.join(CFG['data_dir'], 'dataset_val.pkl'))
dataset_test = load_pickle(os.path.join(CFG['data_dir'], 'dataset_test.pkl'))

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

trainer_1 = deepcopy(trainer)
print('---> Training on wildlife data')
trainer_1.fit(train_dataset=dataset_train, val_dataset=dataset_val)
print('---> Predicting on test data')
preds_bbox = trainer_1.predict(dataset_test)
preds_imgs = map_preds_to_img(
    preds=preds_bbox,
    bbox_keys=dataset_test.keys,
    mapping_dict=dataset_test.bbox_map,
    detector_dict=dataset_test.detector_dict,
)
preds_onehot = {k: np.argmax(v) for k, v in preds_imgs.items()}
preds_empty = set([k for k, v in preds_onehot.items() if v == empty_class])
preds_nonempty = set(preds_onehot.keys()) - set(preds_empty)

tn = len(true_empty.intersection(set(preds_empty)))
tp = len(true_nonempty.intersection(set(preds_nonempty)))
fn = len(true_nonempty.intersection(set(preds_empty)))
fp = len(true_empty.intersection(set(preds_nonempty)))

results_empty_ppl = {
    'tnr': tn / (tn + fp),
    'tpr': tp / (tp + fn),
    'fnr': fn / (tp + fn),
    'fpr': fp / (tn + fp)
}

# PERFORMANCE --------------------------------------------------------------------------

trainer_2 = deepcopy(trainer)
evaluator = Evaluator(
    label_file_path=os.path.join(CFG['data_dir'], CFG['label_file']),
    detector_file_path=os.path.join(CFG['data_dir'], CFG['detector_file']),
    dataset=dataset_test,
    num_classes=trainer_2.get_num_classes(),

)

print('---> Training on wildlife data')
trainer_2.fit(train_dataset=dataset_train, val_dataset=dataset_val)
print('---> Evaluating on test data')
results_perf = evaluator.evaluate(model=trainer_2.get_model())
breakpoint()

# BENEFIT OF TUNING --------------------------------------------------------------------

trainer_3 = deepcopy(trainer)
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

print('---> Training on wildlife data')
trainer_3.fit(train_dataset=dataset_train, val_dataset=dataset_val)
tuning_trainer.fit(train_dataset=dataset_train, val_dataset=dataset_val)

print('---> Evaluating on test data')
results_tuning = {
    'untuned': evaluator.evaluate(model=trainer_3.get_model()),
    'tuned': evaluator.evaluate(model=tuning_trainer.get_model())
}

# WITHOUT AL ---------------------------------------------------------------------------

dataset_is_train = load_pickle(os.path.join(CFG['data_dir'], 'dataset_is_train.pkl'))
dataset_is_val = load_pickle(os.path.join(CFG['data_dir'], 'dataset_is_val.pkl'))
dataset_oos = load_pickle(os.path.join(CFG['data_dir'], 'dataset_oos.pkl'))

trainer_4 = deepcopy(trainer)

print('---> Training on in-sample data')
trainer_4.fit(train_dataset=dataset_is_train, val_dataset=dataset_is_val)
print('---> Evaluating on test data')
evaluator.dataset = dataset_oos
results_perf_passive = evaluator.evaluate(model=trainer_4.get_model())

# WITH AL ------------------------------------------------------------------------------

dataset_pool = load_pickle(os.path.join(CFG['data_dir'], 'dataset_pool.pkl'))
active_learner = ActiveLearner(
    trainer=deepcopy(trainer),
    pool_dataset=dataset_pool,
    label_file_path=os.path.join(CFG['data_dir'], CFG['label_file']),
    empty_class_id=load_json(os.path.join(
        CFG['data_dir'], 'label_map.json')).get('empty'),
    acquisitor_name='entropy',
    train_size=CFG['splits'][0],
    test_dataset=dataset_test,
    test_logfile_path=os.path.join(CFG['data_dir'], CFG['test_logfile']),
    meta_dict=stations_dict,
)

# BENEFIT OF WARMSTART -----------------------------------------------------------------