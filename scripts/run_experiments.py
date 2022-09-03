"""In-sample results."""

from copy import deepcopy
import numpy as np
import os
import ray
from tensorflow import keras
from typing import Dict, Final
from wildlifeml.data import subset_dataset
from wildlifeml.training.trainer import WildlifeTrainer, WildlifeTuningTrainer
from wildlifeml.training.active import ActiveLearner
from wildlifeml.training.evaluator import Evaluator
from wildlifeml.utils.datasets import (
    separate_empties,
    map_preds_to_img,
    map_bbox_to_img,
    do_stratified_splitting
)
from wildlifeml.utils.io import load_csv, load_json, load_pickle
from wildlifeml.utils.misc import flatten_list

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
true_empty = set([k for k, v in label_dict.items() if v == str(empty_class)])
true_nonempty = set(label_dict.keys()) - set(true_empty)

# Prepare training

dataset_train = load_pickle(os.path.join(CFG['data_dir'], 'dataset_train.pkl'))
dataset_val = load_pickle(os.path.join(CFG['data_dir'], 'dataset_val.pkl'))
dataset_test = load_pickle(os.path.join(CFG['data_dir'], 'dataset_test.pkl'))
dataset_pool = load_pickle(os.path.join(CFG['data_dir'], 'dataset_pool.pkl'))

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

# Compute empty-detection performance of MD stand-alone and for entire pipeline

results_empty = {
    'names': ['ours', 'progressive', 'norouzzadeh'],
    'thresholds': [CFG['md_conf'], THRESH_PROGRESSIVE, THRESH_NOROUZZADEH]
}
tnr_md, tpr_md, fnr_md, fpr_md = [], [], [], []
tnr_ppl, tpr_ppl, fnr_ppl, fpr_ppl = [], [], [], []

for threshold in results_empty['thresholds']:

    # Get imgs that MD classifies as empty
    keys_empty_bbox, keys_nonempty_bbox = separate_empties(
        os.path.join(CFG['data_dir'], CFG['detector_file']), threshold
    )
    keys_empty_img = list(set([map_bbox_to_img(k) for k in keys_empty_bbox]))
    keys_nonempty_img = list(set([map_bbox_to_img(k) for k in keys_nonempty_bbox]))

    # Compute confusion metrics for MD stand-alone
    tn_md = len(true_empty.intersection(set(keys_empty_img)))
    tp_md = len(true_nonempty.intersection(set(keys_nonempty_img)))
    fn_md = len(true_nonempty.intersection(set(keys_empty_img)))
    fp_md = len(true_empty.intersection(set(keys_nonempty_img)))
    tnr_md.append(tn_md / (tn_md + fp_md) if (tn_md + fp_md) > 0 else 0.)
    tpr_md.append(tp_md / (tp_md + fn_md) if (tp_md + fn_md) > 0 else 0.)
    fnr_md.append(fn_md / (tp_md + fn_md) if (tp_md + fn_md) > 0 else 0.)
    fpr_md.append(fp_md / (tn_md + fp_md) if (tn_md + fp_md) > 0 else 0.)

    # Prepare new train and val data according to threshold

    dataset_thresh = subset_dataset(dataset_pool, keys_nonempty_bbox)
    share_train = CFG['splits'][0] / (CFG['splits'][0] + CFG['splits'][1])
    share_val = CFG['splits'][1] / (CFG['splits'][0] + CFG['splits'][1])
    imgs_keys = list(set([map_bbox_to_img(k) for k in dataset_thresh.keys]))
    meta_thresh = deepcopy(stations_dict)
    # TODO fix metadict error (some keys have no entry)
    keys_train, _, keys_val = do_stratified_splitting(
        img_keys=imgs_keys,
        splits=(share_train, 0., share_val),
        meta_dict={k: v for k, v in stations_dict.items() if k in imgs_keys},
        random_state=CFG['random_state']
    )
    dataset_train_thresh = subset_dataset(
        dataset_thresh,
        flatten_list([dataset_thresh.mapping_dict[k] for k in keys_train])
    )
    dataset_val_thresh = subset_dataset(
        dataset_thresh,
        flatten_list([dataset_thresh.mapping_dict[k] for k in keys_val])
    )

    # Compute confusion for entire pipeline

    trainer_1 = deepcopy(trainer)
    print('---> Training on wildlife data')
    trainer_1.fit(train_dataset=dataset_train_thresh, val_dataset=dataset_val_thresh)
    print('---> Evaluating on test data')
    evaluator = Evaluator(
        label_file_path=os.path.join(CFG['data_dir'], CFG['label_file']),
        detector_file_path=os.path.join(CFG['data_dir'], CFG['detector_file']),
        dataset=dataset_test,
        num_classes=trainer_1.get_num_classes(),
        conf_threshold=threshold,
    )
    conf_ppl = evaluator.evaluate().get('conf_empty')
    tnr_ppl.append(conf_ppl.get('tnr'))
    tpr_ppl.append(conf_ppl.get('tpr'))
    fnr_ppl.append(conf_ppl.get('fnr'))
    fpr_ppl.append(conf_ppl.get('fpr'))

results_empty.update(
    {
        'tnr_md': [format(x, '.4f') for x in tnr_md],
        'tpr_md': [format(x, '.4f') for x in tpr_md],
        'fnr_md': [format(x, '.4f') for x in fnr_md],
        'fpr_md': [format(x, '.4f') for x in fpr_md],
        'tnr_ppl': [format(x, '.4f') for x in tnr_ppl],
        'tpr_ppl': [format(x, '.4f') for x in tpr_ppl],
        'fnr_ppl': [format(x, '.4f') for x in fnr_ppl],
        'fpr_ppl': [format(x, '.4f') for x in fpr_ppl],
    }
)

print(results_empty)
exit()

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