"""Out-of-sample results."""

import numpy as np
import os
from typing import Dict, Final, List
from wildlifeml.utils.datasets import separate_empties, map_preds_to_img
from wildlifeml.utils.io import load_csv, load_json, load_pickle

CFG: Final[Dict] = load_json(
    '/home/wimmerl/projects/wildlife-experiments/configs/cfg_insample.json'
)

# WITHOUT AL ---------------------------------------------------------------------------

dataset_is_train = load_pickle(os.path.join(CFG['data_dir'], 'dataset_is_train.pkl'))
dataset_is_val = load_pickle(os.path.join(CFG['data_dir'], 'dataset_is_val.pkl'))
dataset_oos = load_pickle(os.path.join(CFG['data_dir'], 'dataset_oos.pkl'))

trainer = load_pickle(os.path.join(CFG['data_dir'], 'trainer.pkl'))
evaluator = load_pickle(os.path.join(CFG['data_dir'], 'evaluator.pkl'))

print('---> Training on in-sample data')
trainer.fit(train_dataset=dataset_is_train, val_dataset=dataset_is_val)
print('---> Evaluating on test data')
evaluator.dataset = dataset_oos
results_perf = evaluator.evaluate(model=trainer.get_model())

# WITH AL ------------------------------------------------------------------------------

# BENEFIT OF WARMSTART -----------------------------------------------------------------