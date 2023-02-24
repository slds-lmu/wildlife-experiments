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
from wildlifeml.utils.datasets import separate_empties
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


TIMESTR: Final[str] = time.strftime("%Y%m%d%H%M")
THRESH_TUNED: Final[float] = 0.1
THRESH_PROGRESSIVE: Final[float] = 0.5
THRESH_NOROUZZADEH: Final[float] = 0.9
BACKBONE_TUNED: Final[str] = 'inception_resnet_v2'
FTLAYERS_TUNED: Final[int] = 0


@click.command()
@click.option(
    '--repo_dir', '-p', help='Your personal path to this repo.', required=True
)
@click.option('--experiment', '-e', help='Experiment to be run.', required=True)
def main(repo_dir: str, experiment: str):

    # ----------------------------------------------------------------------------------
    # GLOBAL ---------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    cfg: Final[Dict] = load_json(os.path.join(repo_dir, 'configs/cfg.json'))
    os.makedirs(cfg['result_dir'], exist_ok=True)
    os.environ['PYTHONHASHSEED'] = str(cfg['random_state'])

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
    trainer_args: Dict = {
        'batch_size': cfg['batch_size'],
        'loss_func': keras.losses.SparseCategoricalCrossentropy(),
        'num_classes': cfg['num_classes'],
        'transfer_epochs': cfg['transfer_epochs'],
        'finetune_epochs': cfg['finetune_epochs'],
        'transfer_optimizer': Adam(cfg['transfer_learning_rate']),
        'finetune_optimizer': Adam(cfg['finetune_learning_rate']),
        'finetune_layers': FTLAYERS_TUNED,
        'model_backbone': BACKBONE_TUNED,
        'transfer_callbacks': transfer_callbacks,
        'finetune_callbacks': finetune_callbacks,
        'num_workers': cfg['num_workers'],
        'eval_metrics': cfg['eval_metrics'],
    }
    evaluator_args: Dict = {
        'label_file_path': os.path.join(cfg['data_dir'], cfg['label_file']),
        'detector_file_path': os.path.join(cfg['data_dir'], cfg['detector_file']),
        'num_classes': cfg['num_classes'],
        'empty_class_id': empty_class_id,
    }

    # ----------------------------------------------------------------------------------
    # PASSIVE LEARNING -----------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    if experiment == 'passive':

        # thresholds = [THRESH_TUNED, THRESH_PROGRESSIVE, THRESH_NOROUZZADEH]
        thresholds = [THRESH_TUNED]
        sample_sizes: Dict = {}
        details_ins_test: Dict = {}
        details_ins_val: Dict = {}

        for threshold in thresholds:

            # Get imgs that MD classifies as empty
            _, keys_all_nonempty = separate_empties(
                os.path.join(cfg['data_dir'], cfg['detector_file']), float(threshold)
            )
            keys_is_train = list(
                set(dataset_is_train.keys).intersection(set(keys_all_nonempty))
            )
            keys_is_val = list(
                set(dataset_is_val.keys).intersection(set(keys_all_nonempty))
            )
            dataset_train_thresh = subset_dataset(dataset_is_train, keys_is_train)
            dataset_val_thresh = subset_dataset(dataset_is_val, keys_is_val)

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

            wandb.init(
                project='wildlilfe',
                tags=[
                    f'conf_{threshold}',
                    BACKBONE_TUNED,
                    f'ftlayers_{FTLAYERS_TUNED}'
                ]
            )
            transfer_callbacks.append(WandbCallback(save_code=True, save_model=False))
            # Compute confusion for entire pipeline
            np.random.seed(cfg['random_state'])
            trainer = WildlifeTrainer(**trainer_args)
            print('---> Training on wildlife data')
            tf.compat.v1.set_random_seed(cfg['random_state'])
            session_conf = tf.compat.v1.ConfigProto(
                intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
            )
            tf.compat.v1.keras.backend.set_session(
                tf.compat.v1.Session(
                    graph=tf.compat.v1.get_default_graph(), config=session_conf
                )
            )
            trainer.fit(
                train_dataset=dataset_train_thresh, val_dataset=dataset_val_thresh
            )
            wandb.finish()
            print('---> Evaluating on in-sample test data')
            evaluator_ins_test = Evaluator(
                dataset=dataset_is_test,
                conf_threshold=float(threshold),
                **evaluator_args,
            )
            evaluator_ins_test.evaluate(trainer)
            details_ins_test[threshold] = evaluator_ins_test.get_details()
            print('---> Evaluating on in-sample val data')
            evaluator_ins_val = Evaluator(
                dataset=dataset_is_val,
                conf_threshold=float(threshold),
                **evaluator_args,
            )
            evaluator_ins_val.evaluate(trainer)
            details_ins_val[threshold] = evaluator_ins_val.get_details()

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
                    os.path.join(cfg['result_dir'], f'{TIMESTR}_oosample.pkl')
                )

        save_as_pickle(
            details_ins_test,
            os.path.join(cfg['result_dir'], f'{TIMESTR}_insample_test.pkl')
        )
        save_as_pickle(
            details_ins_val,
            os.path.join(cfg['result_dir'], f'{TIMESTR}_insample_val.pkl')
        )
        save_as_pickle(
            sample_sizes,
            os.path.join(cfg['result_dir'], f'{TIMESTR}_sample_sizes.pkl')
        )

    # WITH AL (WARM- AND COLDSTART) ----------------------------------------------------

    elif experiment == 'active':

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
        dataset_oos_trainval = subset_dataset(
            dataset_oos_trainval,
            list(set(dataset_oos_trainval.keys).intersection(set(keys_all_nonempty)))
        )

        # Get perf upper limit by training on all data
        print('---> Training on out-of-sample data')
        trainer_al_optimal = WildlifeTrainer(**trainer_args)
        tf.random.set_seed(cfg['random_state'])
        trainer_al_optimal.fit(
            train_dataset=dataset_oos_train, val_dataset=dataset_oos_val
        )
        print('---> Evaluating on out-of-sample data')
        evaluator_al_optimal = Evaluator(
            dataset=dataset_oos_test, conf_threshold=THRESH_TUNED, **evaluator_args
        )
        evaluator_al_optimal.evaluate(trainer_al_optimal)
        details_al_optimal = evaluator_al_optimal.get_details()
        save_as_pickle(
            details_al_optimal,
            os.path.join(cfg['result_dir'], f'{TIMESTR}_results_active_optimal.pkl')
        )

        # Pre-train for warm start
        trainer_pretraining = WildlifeTrainer(**trainer_args)
        trainer_pretraining.finetune_callbacks = finetune_callbacks + [
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(cfg['data_dir'], cfg['pretraining_ckpt']),
                save_weights_only=True,
            )
        ]
        tf.random.set_seed(cfg['random_state'])
        trainer_pretraining.fit(
            load_pickle(os.path.join(cfg['data_dir'], 'dataset_is_train_thresh.pkl')),
            load_pickle(os.path.join(cfg['data_dir'], 'dataset_is_val_thresh.pkl'))
        )

        trainer_args['num_workers'] = 1  # avoid overload due to TF multi-processing
        trainer_args_pretraining: Dict = dict(
            {
                'pretraining_checkpoint': os.path.join(
                    cfg['data_dir'], cfg['pretraining_ckpt']
                )
            },
            **trainer_args
        )
        # Compute batch sizes
        n_obs = len(dataset_oos_trainval.keys)
        init_sizes: Final[List] = [128, 256, 512]
        init_rep: Final[int] = 5
        n_init_batches = sum([x * init_rep for x in init_sizes])
        n_max_batches = (n_obs - n_init_batches) // 1024
        size_last_batch = n_obs - (n_init_batches + n_max_batches * 1024)
        batch_sizes = init_rep * init_sizes + n_max_batches * [1024] + [size_last_batch]

        for args, mode in zip(
                [trainer_args_pretraining, trainer_args], ['warmstart', 'coldstart']
                # [trainer_args], ['coldstart']
        ):
            trainer = WildlifeTrainer(**args)
            active_learner = ActiveLearner(
                trainer=trainer,
                pool_dataset=dataset_oos_trainval,
                label_file_path=os.path.join(cfg['data_dir'], cfg['label_file']),
                empty_class_id=empty_class_id,
                acquisitor_name='entropy',
                train_size=cfg['splits'][0],
                conf_threshold=THRESH_TUNED,
                test_dataset=dataset_oos_test,
                test_logfile_path=os.path.join(
                    cfg['result_dir'], cfg['test_logfile'] + f'_{mode}.pkl'
                ),
                acq_logfile_path=os.path.join(
                    cfg['result_dir'], 'acq_logfile_' + f'{mode}.json'
                ),
                meta_dict=stations_dict,
                active_directory=cfg['active_dir'],
                state_cache=os.path.join(cfg['active_dir'], '.activecache.json'),
                al_batch_size=batch_sizes[0]
            )

            print('---> Running initial AL iteration')
            if os.path.exists(os.path.join(cfg['active_dir'], '.activecache.json')):
                os.remove(os.path.join(cfg['active_dir'], '.activecache.json'))
            tf.random.set_seed(cfg['random_state'])
            active_learner.run()
            active_learner.do_fresh_start = False

            # Set AL iterations to maximum or as specified in config
            if cfg['al_iterations'] < 0:
                al_iterations = len(batch_sizes) - 1
            else:
                al_iterations = min(cfg['al_iterations'], len(batch_sizes) - 1)

            for i in range(al_iterations):
                print(f'---> Starting AL iteration {i + 1}/{al_iterations + 1}')
                keys_to_label = [
                    k for k, _ in load_csv(
                        os.path.join(cfg['active_dir'], 'active_labels.csv')
                    )
                ]
                save_as_csv(
                    [(k, v) for k, v in label_dict.items() if k in keys_to_label],
                    os.path.join(cfg['active_dir'], 'active_labels.csv')
                )
                print('---> Supplied fresh labeled data')
                tf.random.set_seed(cfg['random_state'])
                active_learner.al_batch_size = batch_sizes[i + 1]
                active_learner.run()
                tf.keras.backend.clear_session()
                gc.collect()

            results = load_pickle(active_learner.test_logfile_path)
            results.update({'batch_sizes': batch_sizes})
            save_as_pickle(
                results,
                os.path.join(
                    cfg['result_dir'],
                    f'{TIMESTR}_results_active_{mode}.pkl'
                )
            )
    else:
        raise IOError('Unknown experiment')


if __name__ == '__main__':
    main()
