"""Perform hyper-tuning experiments."""

from typing import (Final, Dict)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TUNE_RESULT_DELIM'] = '/'
import albumentations as A
from tensorflow import keras
from wildlifeml.data import WildlifeDataset, modify_dataset
from wildlifeml.utils.datasets import do_stratified_splitting
from wildlifeml.utils.io import load_csv, load_json
from wildlifeml.training.tuner import WildlifeTuner
import ray



CFG: Final[Dict] = {
    'img_dir': '/common/bothmannl/wildlife_images/usecase2/original_images/',
    'root_dir': '/home/charrakho/Git/wildlife-experiments/data/',
    'label_file': 'labels.csv',
    'detector_file': 'images_megadetector.json',
    'mapping_file': 'bbox_map.json',
    'meta_file': 'stations.csv',
    'splits': (0.7, 0.15, 0.15),
    'local_dir': './ray_results/',
    'random_state': 123,
    'num_classes': 8,
    'transfer_epochs': 0,
    'finetune_epochs': 1,
    'finetune_layers': 1,
    'eval_metrics': ['accuracy'],
    'objective': 'val_accuracy',
    'mode': 'max',
    'n_trials': 2,
    'epochs_per_trial': 1,
    'time_budget': 3600,
    'num_workers': 16,
    'resources_per_trial': {'cpu': 8},
    'max_concurrent_trials': 2,
    'verbose': 0,
    'search_alg_id': 'hyperoptsearch', 
    'scheduler_alg_id':'ashascheduler', 
    'search_space': {
        'model_backbone': ray.tune.choice(['resnet50']),
        'transfer_learning_rate': ray.tune.loguniform(1e-4, 1e-1),
        'finetune_learning_rate': ray.tune.loguniform(1e-4, 1e-1),
        'batch_size': ray.tune.randint(20, 50)},
    'folds': 5,
    'n_runs': 1, 
    'patience': 3,
    'max_epochs': 5,
}

def main() -> None:
    """Perform hyper-tuning experiments."""

    label_dict = {
        k: v for k, v in load_csv(os.path.join(CFG['root_dir'], CFG['label_file']))
    }
    
    stations_dict = {
        k: {'station': v}
        for k, v in load_csv(os.path.join(CFG['root_dir'], CFG['meta_file']))
    }

    mapping_dict = load_json(os.path.join(CFG['root_dir'], CFG['mapping_file']))

    augmentation = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ]
    )

    dataset = WildlifeDataset(
        keys = list(label_dict.keys()), 
        image_dir = CFG['img_dir'], 
        detector_file_path = os.path.join(CFG['root_dir'], CFG['detector_file']),
        label_file_path = os.path.join(CFG['root_dir'], CFG['label_file']),
        bbox_map = mapping_dict,
        batch_size = 1,
        augmentation = augmentation)

    print('---> Split data into train, val and test sets')

    keys_train, keys_val, keys_test = do_stratified_splitting(
        mapping_dict = mapping_dict,
        img_keys = list(label_dict.keys()),
        splits = CFG['splits'],
        meta_dict = stations_dict,
        random_state = CFG['random_state'])
    
    dataset_train = modify_dataset(dataset=dataset, keys=keys_train)
    dataset_val = modify_dataset(dataset=dataset, keys=keys_val)
    #dataset_test = modify_dataset(dataset=dataset, keys=keys_test)
    
    tuner = WildlifeTuner(
        search_space = CFG['search_space'], 
        loss_func = keras.losses.SparseCategoricalCrossentropy(),
        num_classes = CFG['num_classes'],
        transfer_epochs = CFG['transfer_epochs'],
        finetune_epochs = CFG['finetune_epochs'],
        finetune_layers = CFG['finetune_layers'],
        num_workers = CFG['num_workers'],
        eval_metrics = CFG['eval_metrics'],
        local_dir = CFG['local_dir'],
        random_state = CFG['random_state'],
        resources_per_trial = CFG['resources_per_trial'],
        max_concurrent_trials = CFG['max_concurrent_trials'],
        objective = CFG['objective'],
        mode = CFG['mode'],
        n_trials = CFG['n_trials'],
        epochs_per_trial = CFG['epochs_per_trial'],
        time_budget = CFG['time_budget'],
        verbose = CFG['verbose'],
        search_alg_id = CFG['search_alg_id'], 
        scheduler_alg_id = CFG['scheduler_alg_id'],
    )

    tuner.search(
        dataset_train=dataset_train, 
        dataset_val=dataset_val)
    print('---> Completed the searching procedure among different trials.')
    
    print('---> Find the optimal number of epochs for the best trial.')
    tuner.cal_epochs(
        dataset_train = dataset_train, 
        dataset_val = dataset_val, 
        experiment_dir = 'infer', 
        trial_id = 'infer', 
        folds = CFG['folds'], 
        n_runs = CFG['n_runs'], 
        patience = CFG['patience'], 
        max_epochs = CFG['max_epochs'])
    
    print('---> Optimal hyperparameters:, ', tuner.optimal.hps)

if __name__ == '__main__':
    main()