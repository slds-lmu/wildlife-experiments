"""In-sample results."""

import time

import click
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import gc
from typing import Dict, Final, List
from wildlifeml.data import subset_dataset
from wildlifeml.training.trainer import WildlifeTrainer
from wildlifeml.training.active import ActiveLearner
from wildlifeml.training.evaluator import Evaluator
from wildlifeml.utils.datasets import separate_empties, map_bbox_to_img
from wildlifeml.utils.io import (
    load_csv,
    load_json,
    load_pickle,
    save_as_csv,
    save_as_pickle,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import wandb
from wandb.keras import WandbCallback

from utils import seed_everything, MyEarlyStopping

TIMESTR: Final[str] = time.strftime("%Y%m%d%H%M")
THRESH_TUNED: Final[float] = 0.5
THRESH_PROGRESSIVE: Final[float] = 0.5
THRESH_NOROUZZADEH: Final[float] = 0.9
BACKBONE_TUNED: Final[str] = 'xception'
FTLAYERS_TUNED: Final[int] = 0


@click.command()
@click.option(
    '--repo_dir', '-p', help='Your personal path to this repo.', required=True
)
@click.option('--experiment', '-e', help='Experiment to be run.', required=True)
@click.option(
    '--random_seed', '-s', help='Random seed.', required=True
)
@click.option(
    '--acq_criterion',
    '-a',
    help='AL acquisition criterion.',
    required=False,
    default='entropy'
)
def main(repo_dir: str, experiment: str, random_seed: int, acq_criterion: str):

    # ----------------------------------------------------------------------------------
    # GLOBAL ---------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    cfg: Final[Dict] = load_json(os.path.join(repo_dir, 'configs/cfg.json'))
    seed_everything(random_seed)
    os.makedirs(cfg['result_dir'], exist_ok=True)

    # Get metadata
    label_dict = {
        k: v
        for k, v in load_csv(os.path.join(cfg['data_dir'], cfg['label_file']))
    }
    stations_dict = {
        k: {'station': v}
        for k, v in load_csv(os.path.join(cfg['data_dir'], 'stations.csv'))
    }

    # Get data
    dataset_is_train = load_pickle(os.path.join(
        cfg['data_dir'], 'dataset_is_train.pkl')
    )
    dataset_is_val = load_pickle(os.path.join(
        cfg['data_dir'], 'dataset_is_val.pkl')
    )
    dataset_is_test = load_pickle(
        os.path.join(cfg['data_dir'], 'dataset_is_test.pkl')
    )
    dataset_oos_train = load_pickle(
        os.path.join(cfg['data_dir'], 'dataset_oos_train.pkl')
    )
    dataset_oos_val = load_pickle(
        os.path.join(cfg['data_dir'], 'dataset_oos_val.pkl')
    )
    dataset_oos_trainval = load_pickle(os.path.join(
        cfg['data_dir'], 'dataset_oos_trainval.pkl')
    )
    dataset_oos_test = load_pickle(os.path.join(
        cfg['data_dir'], 'dataset_oos_test.pkl')
    )
    for ds in [dataset_is_val, dataset_is_test, dataset_oos_val, dataset_oos_test]:
        ds.shuffle = False
        ds.augmentation = None
    empty_class_id = load_json(
        os.path.join(cfg['data_dir'], 'label_map.json')
    ).get('empty')

    trainer_args: Final[Dict] = {
        'batch_size': cfg['batch_size'],
        'loss_func': keras.losses.SparseCategoricalCrossentropy(),
        'num_classes': cfg['num_classes'],
        'transfer_epochs': cfg['transfer_epochs'],
        'finetune_epochs': cfg['finetune_epochs'],
        'finetune_layers': FTLAYERS_TUNED,
        'model_backbone': BACKBONE_TUNED,
        'num_workers': cfg['num_workers'],
        'eval_metrics': cfg['eval_metrics'],
    }
    evaluator_args: Final[Dict] = {
        'label_file_path': os.path.join(cfg['data_dir'], cfg['label_file']),
        'detector_file_path': os.path.join(cfg['data_dir'], cfg['detector_file']),
        'num_classes': cfg['num_classes'],
        'empty_class_id': empty_class_id,
    }

    # ----------------------------------------------------------------------------------
    # PASSIVE LEARNING -----------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    if experiment == 'passive':

        thresholds = np.arange(0.1, 1, 0.2).round(2).tolist()

        for threshold in thresholds:

            str_thresh = str(int(100 * threshold))
            os.makedirs(
                os.path.join(cfg['result_dir'], 'passive', str_thresh), exist_ok=True
            )

            # Get imgs that MD classifies as empty
            if threshold == 0.:
                keys_is_train = dataset_is_train.keys
                keys_is_val = dataset_is_val.keys
            else:
                _, keys_all_nonempty = separate_empties(
                    os.path.join(
                        cfg['data_dir'], cfg['detector_file']), float(threshold)
                )
                keys_is_train = list(
                    set(dataset_is_train.keys).intersection(set(keys_all_nonempty))
                )
                keys_is_val = list(
                    set(dataset_is_val.keys).intersection(set(keys_all_nonempty))
                )
            dataset_train_thresh = subset_dataset(dataset_is_train, keys_is_train)
            dataset_val_thresh = subset_dataset(dataset_is_val, keys_is_val)
            dataset_test_thresh = subset_dataset(dataset_is_test, dataset_is_test.keys)

            if threshold == 0.:
                # Effectively omit MD from pipeline
                dataset_train_thresh.do_cropping = False
                dataset_val_thresh.do_cropping = False
                dataset_test_thresh.do_cropping = False

            # Save train/val with chosen split for pretraining in active learning
            if threshold == THRESH_TUNED:
                save_as_pickle(
                    dataset_train_thresh,
                    os.path.join(cfg['data_dir'], f'dataset_is_train_thresh.pkl')
                )
                save_as_pickle(
                    dataset_val_thresh,
                    os.path.join(cfg['data_dir'], f'dataset_is_val_thresh.pkl')
                )

            # Prepare training
            transfer_callbacks = [
                EarlyStopping(
                    monitor=cfg['earlystop_metric'],
                    patience=2 * cfg['transfer_patience'],
                ),
                ReduceLROnPlateau(
                    monitor=cfg['earlystop_metric'],
                    patience=cfg['transfer_patience'],
                    factor=0.1,
                    verbose=1,
                ),
            ]
            finetune_callbacks = [
                EarlyStopping(
                    monitor=cfg['earlystop_metric'],
                    patience=2 * cfg['finetune_patience'],
                ),
                ReduceLROnPlateau(
                    monitor=cfg['earlystop_metric'],
                    patience=cfg['finetune_patience'],
                    factor=0.1,
                    verbose=1,
                )
            ]
            wandb.init(
                project='wildlilfe',
                tags=[
                    f'conf_{threshold}',
                    BACKBONE_TUNED,
                    f'ftlayers_{FTLAYERS_TUNED}'
                ]
            )
            transfer_callbacks.append(WandbCallback(save_code=True, save_model=False))
            this_trainer_args: Dict = dict(
                {
                    'transfer_callbacks': transfer_callbacks,
                    'finetune_callbacks': finetune_callbacks,
                    'transfer_optimizer': Adam(cfg['transfer_learning_rate']),
                    'finetune_optimizer': Adam(cfg['finetune_learning_rate']),
                },
                **trainer_args
            )

            trainer = WildlifeTrainer(**this_trainer_args)
            print('---> Training on wildlife data')
            trainer.fit(
                train_dataset=dataset_train_thresh, val_dataset=dataset_val_thresh
            )
            wandb.finish()

            for ds, n in zip(
                [dataset_test_thresh, dataset_val_thresh], ['test', 'val']
            ):
                print('---> Evaluating on in-sample data')
                evaluator = Evaluator(
                    dataset=ds, conf_threshold=float(threshold), **evaluator_args,
                )
                evaluator.evaluate(trainer)
                save_as_pickle(
                    evaluator.get_details(),
                    os.path.join(
                        cfg['result_dir'],
                        'passive',
                        str_thresh,
                        f'{TIMESTR}_insample_{n}_{random_seed}.pkl'
                    )
                )

            if threshold == THRESH_TUNED:
                print('---> Evaluating on out-of-sample test data')
                evaluator_oos = Evaluator(
                    dataset=dataset_oos_test,
                    conf_threshold=float(threshold),
                    **evaluator_args,
                )
                evaluator_oos.evaluate(trainer)
                details_oos = evaluator_oos.get_details()
                save_as_pickle(
                    details_oos,
                    os.path.join(
                        cfg['result_dir'],
                        'passive',
                        str_thresh,
                        f'{TIMESTR}_oosample_{random_seed}.pkl'
                    )
                )

    # WITH AL (WARM- AND COLDSTART) ----------------------------------------------------

    elif experiment == 'active_opt':

        # Prepare OOS data
        _, keys_all_nonempty = separate_empties(
            detector_file_path=os.path.join(cfg['data_dir'], cfg['detector_file']),
            conf_threshold=THRESH_TUNED
        )
        dataset_oos_train = subset_dataset(
            dataset_oos_train,
            list(set(dataset_oos_train.keys).intersection(set(keys_all_nonempty)))
        )
        dataset_oos_val = subset_dataset(
            dataset_oos_val,
            list(set(dataset_oos_val.keys).intersection(set(keys_all_nonempty)))
        )

        # Prepare training
        transfer_callbacks_optimal = [
            EarlyStopping(
                monitor=cfg['earlystop_metric'], patience=2 * cfg['transfer_patience'],
            ),
            ReduceLROnPlateau(
                monitor=cfg['earlystop_metric'],
                patience=cfg['transfer_patience'],
                factor=0.1,
                verbose=1,
            ),
        ]
        finetune_callbacks_optimal = [
            EarlyStopping(
                monitor=cfg['earlystop_metric'],
                patience=2 * cfg['finetune_patience'],
            ),
            ReduceLROnPlateau(
                monitor=cfg['earlystop_metric'],
                patience=cfg['finetune_patience'],
                factor=0.1,
                verbose=1,
            )
        ]
        wandb.init(project='wildlilfe', tags=['active', 'optimal'])
        transfer_callbacks_optimal.append(
            WandbCallback(save_code=True, save_model=False)
        )
        trainer_args_optimal: Dict = dict(
            {
                'transfer_callbacks': transfer_callbacks_optimal,
                'finetune_callbacks': finetune_callbacks_optimal,
                'transfer_optimizer': Adam(cfg['transfer_learning_rate']),
                'finetune_optimizer': Adam(cfg['finetune_learning_rate']),
            },
            **trainer_args
        )
        # Get perf upper limit by training on all data
        trainer_optimal = WildlifeTrainer(**trainer_args_optimal)
        print('---> Training on wildlife data')
        trainer_optimal.fit(
            train_dataset=dataset_oos_train, val_dataset=dataset_oos_val
        )
        wandb.finish()
        print('---> Evaluating on out-of-sample data')
        evaluator = Evaluator(
            dataset=dataset_oos_test,
            conf_threshold=float(THRESH_TUNED),
            **evaluator_args,
        )
        evaluator.evaluate(trainer_optimal)
        save_as_pickle(
            evaluator.get_details(),
            os.path.join(
                cfg['result_dir'],
                'active',
                'optimal',
                f'{TIMESTR}_results_active_optimal_{random_seed}.pkl'
            )
        )

    elif experiment == 'active_pre':

        # Pre-train for warm start
        transfer_callbacks_pretraining = [
            EarlyStopping(
                monitor=cfg['earlystop_metric'], patience=2 * cfg['transfer_patience'],
            ),
            ReduceLROnPlateau(
                monitor=cfg['earlystop_metric'],
                patience=cfg['transfer_patience'],
                factor=0.1,
                verbose=1,
            ),
        ]
        finetune_callbacks_pretraining = [
            EarlyStopping(
                monitor=cfg['earlystop_metric'],
                patience=2 * cfg['finetune_patience'],
            ),
            ReduceLROnPlateau(
                monitor=cfg['earlystop_metric'],
                patience=cfg['finetune_patience'],
                factor=0.1,
                verbose=1,
            )
        ]
        wandb.init(project='wildlilfe', tags=['active', 'pretraining'])
        transfer_callbacks_pretraining.append(
            WandbCallback(save_code=True, save_model=False)
        )
        ckpt_dir = os.path.join(
            cfg['data_dir'], cfg['pretraining_ckpt'], str(random_seed)
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_callback = [
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(ckpt_dir, 'ckpt.hdf5'),
                monitor=cfg['earlystop_metric'],
                mode='min',
                save_weights_only=True,
                save_best_only=True,
            )
        ]
        if FTLAYERS_TUNED == 0 or cfg['finetune_epochs'] == 0:
            transfer_callbacks_pretraining.append(ckpt_callback)
        else:
            finetune_callbacks_pretraining.append(ckpt_callback)
        trainer_args_pretraining: Dict = dict(
            {
                'transfer_callbacks': transfer_callbacks_pretraining,
                'finetune_callbacks': finetune_callbacks_pretraining,
                'transfer_optimizer': Adam(cfg['transfer_learning_rate']),
                'finetune_optimizer': Adam(cfg['finetune_learning_rate']),
            },
            **trainer_args
        )
        trainer_pretraining = WildlifeTrainer(**trainer_args_pretraining)
        seed_everything(random_seed)
        trainer_pretraining.fit(
            load_pickle(os.path.join(cfg['data_dir'], 'dataset_is_train_thresh.pkl')),
            load_pickle(os.path.join(cfg['data_dir'], 'dataset_is_val_thresh.pkl'))
        )
        wandb.finish()

    elif experiment == 'active_exec':

        _, keys_all_nonempty = separate_empties(
            os.path.join(
                cfg['data_dir'], cfg['detector_file']), float(THRESH_TUNED)
        )
        keys_oos_trainval = list(
            set(dataset_oos_trainval.keys).intersection(set(keys_all_nonempty))
        )
        dataset_oos_trainval = subset_dataset(dataset_oos_trainval, keys_oos_trainval)

        # Compute batch sizes
        n_obs = len(map_bbox_to_img(dataset_oos_trainval.keys))
        init_batches: Final[List] = [2**x for x in range(7, 13)]
        batch_sizes: Final[List] = init_batches + [n_obs - sum(init_batches)]

        for mode in ['warmstart', 'coldstart']:

            result_dir = os.path.join(
                cfg['result_dir'], 'active', mode, acq_criterion, str(random_seed)
            )
            os.makedirs(result_dir, exist_ok=True)
            cache_file = os.path.join(
                cfg['active_dir'], str(random_seed), '.activecache.json'
            )
            active_labels_file = os.path.join(
                cfg['active_dir'], str(random_seed), 'active_labels.csv'
            )

            trainer_args_0: Dict = dict(
                {
                    'transfer_optimizer': Adam(cfg['transfer_learning_rate']),
                    'finetune_optimizer': Adam(cfg['finetune_learning_rate']),
                },
                **trainer_args
            )

            active_learner = ActiveLearner(
                trainer=WildlifeTrainer(**trainer_args_0),
                pool_dataset=dataset_oos_trainval,
                label_file_path=os.path.join(
                    cfg['data_dir'], str(random_seed), cfg['label_file']
                ),
                empty_class_id=empty_class_id,
                acquisitor_name=acq_criterion,
                train_size=cfg['splits'][0] / (cfg['splits'][0] + cfg['splits'][1]),
                conf_threshold=THRESH_TUNED,
                test_dataset=dataset_oos_test,
                test_logfile_path=result_dir,
                meta_dict=stations_dict,
                active_directory=os.path.join(cfg['active_dir'], str(random_seed)),
                state_cache=cache_file,
                al_batch_size=batch_sizes[0]
            )

            print('---> Running initial AL iteration')
            if os.path.exists(cache_file):
                os.remove(cache_file)
            seed_everything(random_seed)
            active_learner.run()
            active_learner.do_fresh_start = False

            # Set AL iterations to maximum or as specified in config
            if cfg['al_iterations'] < 0:
                al_iterations = len(batch_sizes)
            else:
                al_iterations = min(cfg['al_iterations'], len(batch_sizes))

            for i in range(al_iterations):

                tf.keras.backend.clear_session()
                tf.compat.v1.reset_default_graph()

                print(f'---> Starting AL iteration {i + 1}/{al_iterations}')
                keys_to_label = [k for k, _ in load_csv(active_labels_file)]
                save_as_csv(
                    [(k, v) for k, v in label_dict.items() if k in keys_to_label],
                    active_labels_file
                )
                print('---> Supplied fresh labeled data')

                wandb.init(
                    project='wildlilfe',
                    tags=['active', f'iter_{i}', mode, acq_criterion]
                )
                trainer_args_i: Dict = dict(
                    {
                        'transfer_callbacks': [
                            MyEarlyStopping(
                                monitor='val_loss',
                                mode='min',
                                patience=2 * cfg['transfer_patience'],
                                start_from_epoch=10,
                                min_delta=0.02,
                            ),
                            ReduceLROnPlateau(
                                monitor=cfg['earlystop_metric'],
                                patience=cfg['transfer_patience'],
                                factor=0.1,
                            ),
                            WandbCallback(save_code=True, save_model=False)
                        ],
                        'transfer_optimizer': Adam(cfg['transfer_learning_rate']),
                        'finetune_optimizer': Adam(cfg['finetune_learning_rate']),
                    },
                    **trainer_args
                )
                if mode == 'warmstart':
                    trainer_args_i.update(
                        {
                            'pretraining_checkpoint': os.path.join(
                                cfg['data_dir'],
                                cfg['pretraining_ckpt'],
                                str(random_seed),
                                'ckpt.hdf5'
                            )
                        }
                    )
                active_learner.set_trainer(WildlifeTrainer(**trainer_args_i))
                if i < al_iterations - 1:
                    batch_size_i = batch_sizes[i + 1]
                    print(f'---> Setting batch size to {batch_size_i}')
                    active_learner.set_batch_size(batch_size_i)
                else:
                    active_learner.set_final()

                seed_everything(random_seed)
                active_learner.run()
                wandb.finish()
                gc.collect()

    else:
        raise IOError('Unknown experiment')


if __name__ == '__main__':
    main()
