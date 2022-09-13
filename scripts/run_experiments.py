"""In-sample results."""

from copy import deepcopy
import os
import ray
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
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
from wildlifeml.utils.io import (
    load_csv,
    load_json,
    load_pickle,
    save_as_json,
    save_as_csv
)
from wildlifeml.utils.misc import flatten_list

CFG: Final[Dict] = load_json(
    '/home/wimmerl/projects/wildlife-experiments/configs/cfg_insample.json'
)
RESULT_DIR: Final[str] = '/home/wimmerl/projects/wildlife-experiments/results/'
N_GPU = len(os.environ['CUDA_VISIBLE_DEVICES'])
N_CPU: Final[int] = 16

os.makedirs(RESULT_DIR, exist_ok=True)

# EMPTY VS NON-EMPTY -------------------------------------------------------------------

THRESH_PROGRESSIVE: Final[float] = 0.5
THRESH_NOROUZZADEH: Final[float] = 0.9

# Get metadata

detector_dict = load_json(os.path.join(CFG['data_dir'], CFG['detector_file']))
label_dict = {
    k: v for k, v in load_csv(os.path.join(CFG['data_dir'], CFG['label_file']))
}
stations_dict = {
    k: {'station': v}
    for k, v in load_csv(os.path.join(CFG['data_dir'], CFG['meta_file']))
}
# Get truly empty images

empty_class = load_json(os.path.join(CFG['data_dir'], 'label_map.json')).get('empty')
true_empty = set([k for k, v in label_dict.items() if v == str(empty_class)])
true_nonempty = set(label_dict.keys()) - set(true_empty)

# Prepare training

dataset_is_train = load_pickle(os.path.join(CFG['data_dir'], 'dataset_is_train.pkl'))
dataset_is_val = load_pickle(os.path.join(CFG['data_dir'], 'dataset_is_val.pkl'))
dataset_is_trainval = load_pickle(os.path.join(
    CFG['data_dir'], 'dataset_is_trainval.pkl')
)
dataset_is_test = load_pickle(os.path.join(CFG['data_dir'], 'dataset_is_test.pkl'))

trainer = WildlifeTuningTrainer(
    search_space={
        'backbone': ray.tune.choice(['resnet50']),
        'transfer_learning_rate': ray.tune.choice([1e-4]),
        'finetune_learning_rate': ray.tune.choice([1e-4]),
        'batch_size': ray.tune.choice([32])
    },
    loss_func=keras.losses.SparseCategoricalCrossentropy(),
    num_classes=CFG['num_classes'],
    transfer_epochs=CFG['transfer_epochs'],
    finetune_epochs=CFG['finetune_epochs'],
    transfer_optimizer=Adam(),
    finetune_optimizer=Adam(),
    finetune_layers=CFG['finetune_layers'],
    transfer_callbacks=None,
    finetune_callbacks=None,
    num_workers=CFG['num_workers'],
    eval_metrics=CFG['eval_metrics'],
    resources_per_trial={'cpu': 4, 'gpu': N_GPU},
    max_concurrent_trials=1,
    time_budget=60,
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
    keys_empty_bbox = list(
        set(keys_empty_bbox).intersection(set(dataset_is_trainval.keys))
    )
    keys_nonempty_bbox = list(
        set(keys_nonempty_bbox).intersection(set(dataset_is_trainval.keys))
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

    dataset_thresh = subset_dataset(dataset_is_trainval, keys_nonempty_bbox)
    share_train = CFG['splits'][0] / (CFG['splits'][0] + CFG['splits'][1])
    share_val = CFG['splits'][1] / (CFG['splits'][0] + CFG['splits'][1])
    imgs_keys = list(set([map_bbox_to_img(k) for k in dataset_thresh.keys]))
    meta_thresh = deepcopy(stations_dict)
    for k in (set(imgs_keys) - set(meta_thresh)):
        meta_thresh.update({k: {'station': None}})
    keys_train, _, keys_val = do_stratified_splitting(
        img_keys=imgs_keys,
        splits=(share_train, 0., share_val),
        meta_dict=meta_thresh,
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

    trainer_empty = deepcopy(trainer)
    print('---> Training on wildlife data')
    trainer_empty.fit(
        train_dataset=dataset_train_thresh, val_dataset=dataset_val_thresh
    )
    print('---> Evaluating on test data')
    evaluator = Evaluator(
        label_file_path=os.path.join(CFG['data_dir'], CFG['label_file']),
        detector_file_path=os.path.join(CFG['data_dir'], CFG['detector_file']),
        dataset=dataset_is_test,
        num_classes=trainer_empty.get_num_classes(),
        conf_threshold=threshold,
    )
    conf_ppl = evaluator.evaluate(trainer_empty).get('conf_empty')
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

save_as_json(results_empty, os.path.join(RESULT_DIR, 'results_insample_empty.json'))
exit()

# PERFORMANCE --------------------------------------------------------------------------

trainer_perf_is = deepcopy(trainer)
evaluator = Evaluator(
    label_file_path=os.path.join(CFG['data_dir'], CFG['label_file']),
    detector_file_path=os.path.join(CFG['data_dir'], CFG['detector_file']),
    dataset=dataset_is_test,
    num_classes=trainer_perf_is.get_num_classes(),
)

print('---> Training on wildlife data')
trainer_perf_is.fit(train_dataset=dataset_is_train, val_dataset=dataset_is_val)
print('---> Evaluating on test data')
results_perf = evaluator.evaluate(trainer_perf_is)

save_as_json(results_perf, os.path.join(RESULT_DIR, 'results_insample_perf.json'))
exit()

# # BENEFIT OF TUNING --------------------------------------------------------------------

# TODO: set hyperparams appropriately
trainer_untuned_default = WildlifeTrainer(
    loss_func=keras.losses.SparseCategoricalCrossentropy(),
    num_classes=CFG['num_classes'],
    transfer_epochs=CFG['transfer_epochs'],
    finetune_epochs=CFG['finetune_epochs'],
    transfer_optimizer=Adam(),
    finetune_optimizer=Adam(),
    finetune_layers=CFG['finetune_layers'],
    transfer_callbacks=None,
    finetune_callbacks=None,
    num_workers=CFG['num_workers'],
    eval_metrics=CFG['eval_metrics'],
)
trainer_untuned_random = WildlifeTrainer(
    loss_func=keras.losses.SparseCategoricalCrossentropy(),
    num_classes=CFG['num_classes'],
    transfer_epochs=CFG['transfer_epochs'],
    finetune_epochs=CFG['finetune_epochs'],
    transfer_optimizer=Adam(),
    finetune_optimizer=Adam(),
    finetune_layers=CFG['finetune_layers'],
    transfer_callbacks=None,
    finetune_callbacks=None,
    num_workers=CFG['num_workers'],
    eval_metrics=CFG['eval_metrics'],
)

print('---> Training on wildlife data')
trainer_untuned_default.fit(train_dataset=dataset_is_train, val_dataset=dataset_is_val)
trainer_untuned_random.fit(train_dataset=dataset_is_train, val_dataset=dataset_is_val)

print('---> Evaluating on test data')
results_tuning = {
    'untuned_default': evaluator.evaluate(trainer_untuned_default),
    'untuned_random': evaluator.evaluate(trainer_untuned_random),
    'tuned': results_perf,
}

save_as_json(results_tuning, os.path.join(RESULT_DIR, 'results_insample_tuning.json'))
exit()

# WITHOUT AL ---------------------------------------------------------------------------

dataset_is_train = load_pickle(os.path.join(CFG['data_dir'], 'dataset_is_train.pkl'))
dataset_is_val = load_pickle(os.path.join(CFG['data_dir'], 'dataset_is_val.pkl'))
dataset_oos = load_pickle(os.path.join(CFG['data_dir'], 'dataset_oos.pkl'))

trainer_perf_oos = deepcopy(trainer)

print('---> Training on in-sample data')
trainer_perf_oos.fit(train_dataset=dataset_is_train, val_dataset=dataset_is_val)
print('---> Evaluating on test data')
evaluator.dataset = dataset_oos
results_perf_passive = evaluator.evaluate(trainer_perf_oos)

save_as_json(
    results_perf_passive, os.path.join(RESULT_DIR, 'results_oosample_perf.json')
)
exit()

# # WITH AL ------------------------------------------------------------------------------

dataset_is = load_pickle(os.path.join(CFG['data_dir'], 'dataset_is.pkl'))

# TODO: pretrain model on dataset_is

active_learner = ActiveLearner(
    trainer=deepcopy(trainer),
    pool_dataset=dataset_is,
    label_file_path=os.path.join(CFG['data_dir'], CFG['label_file']),
    empty_class_id=load_json(os.path.join(
        CFG['data_dir'], 'label_map.json')).get('empty'),
    acquisitor_name='entropy',
    train_size=CFG['splits'][0],
    test_dataset=dataset_oos,
    test_logfile_path=os.path.join(CFG['data_dir'], CFG['test_logfile']),
    meta_dict=stations_dict,
    active_directory=CFG['active_dir'],
)

print('---> Running initial AL iteration')
active_learner.run()
active_learner.do_fresh_start = False

for i in range(CFG['al_iterations']):
    print(f'---> Starting AL iteration {i + 1}')
    imgs_to_label = os.listdir(os.path.join(CFG['active_dir'], 'images'))
    labels_supplied = [
        (k, v) for k, v in label_dict.items() if k in imgs_to_label
    ]
    save_as_csv(labels_supplied, os.path.join(CFG['active_dir'], 'active_labels.csv'))
    print('---> Supplied fresh labeled data')
    active_learner.run()

results_active = load_json(active_learner.test_logfile_path)
save_as_json(results_active, os.path.join(RESULT_DIR, 'results_oosample_active.json'))
exit()

# BENEFIT OF WARMSTART -----------------------------------------------------------------

# trainer_coldstart = WildlifeTuningTrainer(
#     search_space={
#         'backbone': ray.tune.choice(['resnet50']),
#         'transfer_learning_rate': ray.tune.choice([1e-4]),
#         'finetune_learning_rate': ray.tune.choice([1e-4]),
#         'batch_size': ray.tune.choice([32])
#     },
#     loss_func=keras.losses.SparseCategoricalCrossentropy(),
#     num_classes=CFG['num_classes'],
#     transfer_epochs=CFG['transfer_epochs'],
#     finetune_epochs=CFG['finetune_epochs'],
#     transfer_optimizer=Adam(),
#     finetune_optimizer=Adam(),
#     finetune_layers=CFG['finetune_layers'],
#     transfer_callbacks=None,
#     finetune_callbacks=None,
#     num_workers=CFG['num_workers'],
#     eval_metrics=CFG['eval_metrics'],
#     resources_per_trial={'cpu': 4, 'gpu': N_GPU},
#     max_concurrent_trials=1,
#     time_budget=60,
# )