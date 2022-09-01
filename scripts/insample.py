"""In-sample results."""

import numpy as np
import os
from typing import Dict, Final
from wildlifeml.utils.datasets import separate_empties, map_preds_to_img
from wildlifeml.utils.io import load_csv, load_json, load_pickle

CFG: Final[Dict] = load_json(
    '/home/wimmerl/projects/wildlife-experiments/configs/cfg_insample.json'
)

# EMPTY VS NON-EMPTY -------------------------------------------------------------------

THRESH_PROGRESSIVE: Final[float] = 0.5
THRESH_NOROUZZADEH: Final[float] = 0.9

# Get metadata

detector_dict = load_json(os.path.join(CFG['data_dir'], CFG['detector_file']))
label_dict = {
    k: v for k, v in load_csv(os.path.join(CFG['data_dir'], CFG['label_file']))
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

for idx, threshold in results_empty_md['thresholds']:
    keys_empty, keys_nonempty = separate_empties(os.path.join(
        CFG['data_dir'], CFG['detector_file']), threshold
    )
    tn = len(true_empty.intersection(set(keys_empty)))
    tp = len(true_nonempty.intersection(set(keys_nonempty)))
    fn = len(true_nonempty.intersection(set(keys_empty)))
    fp = len(true_empty.intersection(set(keys_nonempty)))
    tnr[idx] = tn / (tn + fp)
    tpr[idx] = tp / (tp + fn)
    fnr[idx] = fn / (tp + fn)
    fpr[idx] = fp / (tn + fp)

results_empty_md.update({'tnr': tnr, 'tpr': tpr, 'fnr': fnr, 'fpr': fpr})

# Compute empty-detection performance of pipeline

dataset_train = load_pickle(os.path.join(CFG['data_dir'], 'dataset_train.pkl'))
dataset_val = load_pickle(os.path.join(CFG['data_dir'], 'dataset_val.pkl'))
dataset_test = load_pickle(os.path.join(CFG['data_dir'], 'dataset_test.pkl'))
trainer_1 = load_pickle(os.path.join(CFG['data_dir'], 'trainer.pkl'))

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

trainer_2 = load_pickle(os.path.join(CFG['data_dir'], 'trainer.pkl'))
evaluator = load_pickle(os.path.join(CFG['data_dir'], 'evaluator.pkl'))

print('---> Training on wildlife data')
trainer_2.fit(train_dataset=dataset_train, val_dataset=dataset_val)
print('---> Evaluating on test data')
results_perf = evaluator.evaluate(model=trainer_2.get_model())
breakpoint()

# BENEFIT OF TUNING --------------------------------------------------------------------

trainer_3 = load_pickle(os.path.join(CFG['data_dir'], 'trainer.pkl'))
tuning_trainer = load_pickle(os.path.join(CFG['data_dir'], 'tuning_trainer.pkl'))

print('---> Training on wildlife data')
trainer_3.fit(train_dataset=dataset_train, val_dataset=dataset_val)
tuning_trainer.fit(train_dataset=dataset_train, val_dataset=dataset_val)

print('---> Evaluating on test data')
results_tuning = {
    'untuned': evaluator.evaluate(model=trainer_3.get_model()),
    'tuned': evaluator.evaluate(model=tuning_trainer.get_model())
}