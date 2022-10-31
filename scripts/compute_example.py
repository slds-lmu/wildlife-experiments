"""Examples of how to use the wildlifeml package."""


import albumentations as A
import click
import os
from typing import Dict, Final, List

from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from wildlifeml.preprocessing.megadetector import MegaDetector
from wildlifeml.training.trainer import WildlifeTrainer
from wildlifeml.training.active import ActiveLearner
from wildlifeml.training.evaluator import Evaluator
from wildlifeml.data import BBoxMapper, WildlifeDataset, subset_dataset
from wildlifeml.utils.io import (
    load_csv,
    save_as_csv,
    load_json,
    save_as_json,
    load_pickle,
    save_as_pickle
)
from wildlifeml.utils.datasets import (
    separate_empties,
    do_stratified_splitting,
    map_bbox_to_img
)
from wildlifeml.utils.misc import flatten_list
from wildlifeml.utils.metrics import (
    SparseCategoricalRecall,
    SparseCategoricalPrecision,
    SparseCategoricalF1
)


# Define metrics for evaluating predictions
EVAL_METRICS: Final[List] = [
    'accuracy',
    SparseCategoricalRecall(name='recall'),
    SparseCategoricalPrecision(name='precision'),
    SparseCategoricalF1(name='f1'),
]


@click.command()
@click.option('--config_file', '-cf', help='Path to config file.', required=True)
@click.option('--task', '-tk', help='Task to be run.', required=True)
def main(config_file: str, task: str):

    # Load file with all user-specified configurations
    cfg: Final[Dict] = load_json(config_file)

    # PREPARE DATA ---------------------------------------------------------------------

    if task == 'prep':

        # Prepare datasets for max 3 modi: training data to learn model on, pre-training
        # data to warm-start model (optional), production data to predict on (optional)

        label_files = [cfg['label_file_train'], cfg['label_file_test']]
        modes = ['train', 'test']
        if len(cfg['label_file_pretrain']) > 0:
            label_files.append(cfg['label_file_pretrain'])
            modes.append('pretrain')
        if len(cfg['label_file_prod']) > 0:
            label_files.append(cfg['label_file_prod'])
            modes.append('prod')

        for label_file, mode in zip(label_files, modes):
            # Create label map as a look-up for the class encoding
            label_dict_original = {k: v for k, v in load_csv(label_file)}
            class_names = sorted(list(set(label_dict_original.values())))
            label_map = {class_names[i]: i for i in range(len(class_names))}
            save_as_json(
                label_map, os.path.join(cfg['data_dir'], f'label_map_{mode}.json')
            )
            label_dict = {k: label_map[v] for k, v in label_dict_original.items()}
            save_as_csv(
                [(k, v) for k, v in label_dict.items()],
                os.path.join(cfg['data_dir'], f'label_file_{mode}_num.csv')
            )

            # Run MegaDetector if there is no detection file already present (if such a
            # file is present, and you wish to re-run the MegaDetector anyway, you need
            # to remove or rename the file)
            detector_file_path = os.path.join(
                cfg['data_dir'], cfg[f'detector_file_{mode}']
            )
            if not os.path.exists(detector_file_path):
                md = MegaDetector(
                    batch_size=cfg['md_batchsize'], confidence_threshold=cfg['md_conf']
                )
                md.predict_directory(
                    directory=os.path.join(cfg['img_dir']),
                    output_file=os.path.join(detector_file_path),
                )

            # Create mapping from images to bounding boxes
            mapper = BBoxMapper(detector_file_path)
            bbox_map = mapper.get_keymap()
            save_as_json(
                bbox_map, os.path.join(cfg['data_dir'], f'bbox_map_{mode}.json')
            )

            # Filter non-empty images as detected by MegaDetector
            _, nonempty_keys_md = separate_empties(
                detector_file_path=os.path.join(detector_file_path)
            )
            nonempty_keys = list(
                set(nonempty_keys_md).intersection(
                    set(flatten_list([v for v in bbox_map.values()]))
                )
            )
            # Create dataset
            dataset = WildlifeDataset(
                keys=nonempty_keys,
                image_dir=cfg['img_dir'],
                detector_file_path=detector_file_path,
                label_file_path=os.path.join(
                    cfg['data_dir'], f'label_file_{mode}_num.csv'
                ),
                bbox_map=bbox_map,
                batch_size=cfg['batch_size'],
                augmentation=A.Compose(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.Rotate(p=0.5),
                        A.RandomBrightnessContrast(p=0.2),
                    ]
                ),
            )
            save_as_pickle(
                dataset, os.path.join(cfg['data_dir'], f'dataset_{mode}.pkl')
            )

            if mode == 'train':
                # Split training data in actual training data and validation data.
                # If extra metadata for splitting are provided, create dict (even if
                # not, data are always stratified by class).
                if len(cfg['meta_file_train']) > 0:
                    meta_dict = {
                        k: {'meta_var': v} for k, v in load_csv(cfg['meta_file_train'])
                    }
                else:
                    meta_dict = {k: {'meta_var': v} for k, v in label_dict.items()}
                save_as_json(meta_dict, os.path.join(cfg['data_dir'], 'meta_dict.json'))
                keys_all = list(set(nonempty_keys))
                keys_all_img = [map_bbox_to_img(k) for k in keys_all]
                keys_train, _, keys_val = do_stratified_splitting(
                    img_keys=list(set(keys_all_img)),
                    splits=cfg['splits'],
                    meta_dict=meta_dict,
                    random_state=cfg['random_state']
                )
                for keyset, partition in zip(
                        [keys_train, keys_val], ['train', 'val']):
                    subset = subset_dataset(
                        dataset, flatten_list([dataset.mapping_dict[k] for k in keyset])
                    )
                    save_as_pickle(
                        subset,
                        os.path.join(cfg['data_dir'], f'dataset_t_{partition}.pkl')
                    )
        exit()

    elif task in ['train_passive', 'train_active']:

        # Load training data (without splitting) and test data
        dataset_train = load_pickle(
            os.path.join(cfg['data_dir'], 'dataset_train.pkl')
        )
        dataset_test = load_pickle(
            os.path.join(cfg['data_dir'], 'dataset_test.pkl')
        )

        # Specify training arguments
        trainer_args: Dict = {
            'batch_size': cfg['batch_size'],
            'loss_func': keras.losses.SparseCategoricalCrossentropy(),
            'num_classes': cfg['num_classes'],
            'transfer_epochs': cfg['transfer_epochs'],
            'finetune_epochs': cfg['finetune_epochs'],
            'transfer_optimizer': Adam(learning_rate=cfg['transfer_learning_rate']),
            'finetune_optimizer': Adam(learning_rate=cfg['finetune_learning_rate']),
            'finetune_layers': cfg['finetune_layers'],
            'model_backbone': cfg['model_backbone'],
            'transfer_callbacks': None,
            'finetune_callbacks': None,
            'num_workers': cfg['num_workers'],
            'eval_metrics': EVAL_METRICS,
        }

        # If pre-training data are available, pre-train for warmstart
        if os.path.exists(os.path.join(cfg['data_dir'], 'dataset_pretrain.pkl')):
            dataset_pretrain = load_pickle(
                os.path.join(cfg['data_dir'], 'dataset_pretrain.pkl')
            )
            trainer_pretrain = WildlifeTrainer(**trainer_args)
            trainer_pretrain.finetune_callbacks = keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(cfg['data_dir'], 'pretraining_ckpt'),
                save_weights_only=True,
            )
            trainer_pretrain.fit(train_dataset=dataset_pretrain)
            trainer_args_pretraining = dict(
                {
                    'pretraining_checkpoint': os.path.join(
                        cfg['data_dir'], 'pretraining_ckpt'
                    )
                },
                **trainer_args
            )
            trainer = WildlifeTrainer(**trainer_args_pretraining)

        else:
            trainer = WildlifeTrainer(**trainer_args)

        # Perform standard (passive) learning

        if task == 'train_passive':

            # Load datasets
            dataset_t_train = load_pickle(
                os.path.join(cfg['data_dir'], 'dataset_t_train.pkl')
            )
            dataset_t_val = load_pickle(
                os.path.join(cfg['data_dir'], 'dataset_t_val.pkl')
            )
            # Instantiate evaluator
            evaluator = Evaluator(
                label_file_path=os.path.join(
                    cfg['data_dir'], 'label_file_test_num.csv'
                ),
                detector_file_path=os.path.join(
                    cfg['data_dir'], cfg['detector_file_test']
                ),
                dataset=dataset_test,
                num_classes=cfg['num_classes'],
            )
            # Train
            print('---> Training on wildlife data')
            trainer.fit(train_dataset=dataset_t_train, val_dataset=dataset_t_val)
            # Evaluate
            print('---> Evaluating on test data')
            test_results = evaluator.evaluate(trainer)
            save_as_json(test_results, cfg['result_file'])

            # Predict on production data if desired
            if os.path.exists(os.path.join(cfg['data_dir'], 'dataset_prod.pkl')):
                dataset_prod = load_pickle(
                    os.path.join(cfg['data_dir'], 'dataset_prod.pkl')
                )
                trainer = WildlifeTrainer(**trainer_args)
                print('---> Training on wildlife data')
                trainer.fit(train_dataset=dataset_train)
                print('---> Predicting on production data')
                predictions = trainer.predict(dataset_prod)
                # TODO save predictions

        else:

            active_learner = ActiveLearner(
                trainer=trainer,
                pool_dataset=dataset_train,
                label_file_path=os.path.join(cfg['data_dir'], cfg['label_file_active']),
                empty_class_id=load_json(os.path.join(
                    cfg['data_dir'], 'label_map_train.json')
                ).get('empty'),
                acquisitor_name=cfg['acquisition_function'],
                train_size=cfg['splits'][0],
                test_dataset=dataset_test,
                test_logfile_path=cfg['result_file'],
                meta_dict=load_json(os.path.join(cfg['data_dir'], 'meta_dict.json')),
                active_directory=cfg['active_dir'],
                state_cache=os.path.join(cfg['active_dir'], '.activecache.json')
            )

            print('---> Running initial AL iteration')
            if os.path.exists(os.path.join(cfg['active_dir'], '.activecache.json')):
                os.remove(os.path.join(cfg['active_dir'], '.activecache.json'))
            active_learner.run()
            active_learner.do_fresh_start = False

            for i in range(cfg['al_iterations']):
                print(f'---> Starting AL iteration {i + 1}')
                if not bool(cfg['human_annotation']):
                    keys_to_label = [
                        k for k, _ in load_csv(
                            os.path.join(cfg['active_dir'], 'active_labels.csv')
                        )
                    ]
                    label_dict = {k: v for k, v in load_csv(cfg['label_file_train'])}
                    labels_supplied = [
                        (k, v) for k, v in label_dict.items() if k in keys_to_label
                    ]
                    save_as_csv(
                        labels_supplied,
                        os.path.join(cfg['active_dir'], 'active_labels.csv')
                    )
                    print('---> Supplied fresh labeled data')
                    active_learner.run()

    else:
        raise ValueError(f'Task "{task}" not implemented.')


if __name__ == '__main__':
    main()