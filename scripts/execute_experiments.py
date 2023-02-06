"""In-sample results."""
import itertools
import random
import time
import click
from copy import deepcopy
import os

import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import gc
from typing import Dict, Final, List
from wildlifeml.data import subset_dataset
from wildlifeml.training.trainer import WildlifeTrainer
from wildlifeml.training.active import ActiveLearner
from wildlifeml.training.evaluator import Evaluator
from wildlifeml.utils.datasets import (
    separate_empties,
    map_bbox_to_img,
    do_stratified_splitting
)
from wildlifeml.utils.io import (
    load_csv,
    load_json,
    load_pickle,
    save_as_csv,
    save_as_json,
    save_as_pickle,
)
from wildlifeml.utils.misc import flatten_list
from tensorflow.keras.callbacks import EarlyStopping


TIMESTR: Final[str] = time.strftime("%Y%m%d%H%M")
THRESH_TUNED: Final[float] = 0.1
THRESH_PROGRESSIVE: Final[float] = 0.5
THRESH_NOROUZZADEH: Final[float] = 0.9
BACKBONE_TUNED: Final[str] = 'xception'
FTLAYERS_TUNED: Final[int] = 6

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

    # Get metadata
    label_dict = {
        k: v
        for k, v in load_csv(os.path.join(cfg['data_dir'], cfg['label_file']))
    }
    stations_dict = {
        k: {'station': v}
        for k, v in load_csv(os.path.join(cfg['data_dir'], 'stations.csv'))
    }

    # Prepare training
    dataset_is_train = load_pickle(
        os.path.join(cfg['data_dir'], 'dataset_is_train.pkl')
    )
    dataset_is_val = load_pickle(
        os.path.join(cfg['data_dir'], 'dataset_is_val.pkl')
    )
    dataset_is_trainval = load_pickle(os.path.join(
        cfg['data_dir'], 'dataset_is_trainval.pkl')
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
    # Remove empty images from train/val keys according to MD threshold
    _, keys_all_nonempty = separate_empties(
        detector_file_path=os.path.join(cfg['data_dir'], cfg['detector_file']),
        conf_threshold=THRESH_TUNED
    )
    keys_is_train = list(
        set(dataset_is_train.keys).intersection(set(keys_all_nonempty))
    )
    keys_is_val = list(
        set(dataset_is_val.keys).intersection(set(keys_all_nonempty))
    )
    dataset_is_train = subset_dataset(dataset_is_train, keys_is_train)
    dataset_is_val = subset_dataset(dataset_is_val, keys_is_val)

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
    empty_class_id = load_json(
        os.path.join(cfg['data_dir'], 'label_map.json')
    ).get('empty')

    evaluator_is = Evaluator(
        label_file_path=os.path.join(cfg['data_dir'], cfg['label_file']),
        detector_file_path=os.path.join(cfg['data_dir'], cfg['detector_file']),
        dataset=dataset_is_test,
        num_classes=cfg['num_classes'],
        empty_class_id=empty_class_id,
    )
    evaluator_oos = Evaluator(
        label_file_path=os.path.join(cfg['data_dir'], cfg['label_file']),
        detector_file_path=os.path.join(cfg['data_dir'], cfg['detector_file']),
        dataset=dataset_oos_test,
        num_classes=cfg['num_classes'],
        empty_class_id=empty_class_id,
    )

    # ----------------------------------------------------------------------------------
    # IN-SAMPLE ------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    # PERFORMANCE ----------------------------------------------------------------------

    if experiment == 'insample_perf':
        trainer_perf_is = WildlifeTrainer(**trainer_args)
        print('---> Training on wildlife data')
        tf.random.set_seed(cfg['random_state'])
        trainer_perf_is.fit(
            train_dataset=dataset_is_train,
            val_dataset=dataset_is_val
        )
        print('---> Evaluating on test data')
        evaluator_is.evaluate(trainer_perf_is)
        details_perf = evaluator_is.get_details()
        save_as_pickle(
            details_perf,
            os.path.join(
                cfg['result_dir'],
                f'{TIMESTR}_insample_perf.pickle'
            )
        )

    # EMPTY VS NON-EMPTY ---------------------------------------------------------------

    elif experiment == 'insample_empty':

        thresholds = [cfg['md_conf'], THRESH_PROGRESSIVE, THRESH_NOROUZZADEH]
        details_empty = {}

        for threshold in thresholds:

            # Get imgs that MD classifies as empty
            _, keys_nonempty_bbox = separate_empties(
                os.path.join(cfg['data_dir'], cfg['detector_file']), float(threshold)
            )
            keys_nonempty_bbox_trainval = list(
                set(keys_nonempty_bbox).intersection(set(dataset_is_trainval.keys))
            )

            # Prepare new train and val data according to threshold

            dataset_thresh = subset_dataset(dataset_is_trainval, keys_nonempty_bbox_trainval)
            share_train = cfg['splits'][0] / (cfg['splits'][0] + cfg['splits'][1])
            share_val = cfg['splits'][1] / (cfg['splits'][0] + cfg['splits'][1])
            imgs_keys = list(set([map_bbox_to_img(k) for k in dataset_thresh.keys]))
            meta_thresh = deepcopy(stations_dict)
            for k in (set(imgs_keys) - set(meta_thresh)):
                meta_thresh.update({k: {'station': None}})
            keys_train, _, keys_val = do_stratified_splitting(
                img_keys=imgs_keys,
                splits=(share_train, 0., share_val),
                meta_dict=meta_thresh,
                random_state=cfg['random_state']
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

            trainer_empty = WildlifeTrainer(**trainer_args)
            print('---> Training on wildlife data')
            tf.random.set_seed(cfg['random_state'])
            trainer_empty.fit(
                train_dataset=dataset_train_thresh,
                val_dataset=dataset_val_thresh,
            )
            print('---> Evaluating on test data')
            evaluator = Evaluator(
                label_file_path=os.path.join(cfg['data_dir'], cfg['label_file']),
                detector_file_path=os.path.join(cfg['data_dir'], cfg['detector_file']),
                dataset=dataset_is_test,
                num_classes=trainer_empty.get_num_classes(),
                empty_class_id=empty_class_id,
                conf_threshold=float(threshold),
            )
            evaluator.evaluate(trainer_empty)
            details_empty[threshold] = evaluator.get_details()

        save_as_pickle(
            details_empty,
            os.path.join(
                cfg['result_dir'],
                f'{TIMESTR}_insample_empty.pickle'
            )
        )

    # ----------------------------------------------------------------------------------
    # OUT-OF-SAMPLE --------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    # WITHOUT AL -----------------------------------------------------------------------

    elif experiment == 'oosample_perf':
        print('---> Training on in-sample data')
        trainer_perf_oos = WildlifeTrainer(**trainer_args)
        tf.random.set_seed(cfg['random_state'])
        trainer_perf_oos.fit(
            train_dataset=dataset_is_train,
            val_dataset=dataset_is_val
        )
        print('---> Evaluating on out-of-sample data')
        evaluator_oos.evaluate(trainer_perf_oos)
        details_perf_passive = evaluator_oos.get_details()

        save_as_pickle(
            details_perf_passive,
            os.path.join(
                cfg['result_dir'],
                f'{TIMESTR}_oosample_perf.pickle'
            )
        )

    # WITH AL (WARM- AND COLDSTART) ----------------------------------------------------

    elif experiment == 'oosample_active':

        # Get perf upper limit by training on all data
        print('---> Training on out-of-sample data')
        trainer_al_optimal = WildlifeTrainer(**trainer_args)
        tf.random.set_seed(cfg['random_state'])
        trainer_al_optimal.fit(
            train_dataset=dataset_oos_train, val_dataset=dataset_oos_val
        )
        print('---> Evaluating on out-of-sample data')
        results_al_optimal = evaluator_oos.evaluate(trainer_al_optimal)
        save_as_json(
            results_al_optimal,
            os.path.join(
                cfg['result_dir'],
                f'{TIMESTR}_results_oosample_active_optimal.json'
            )
        )

        # Pre-train for warm start
        trainer_pretraining = WildlifeTrainer(**trainer_args)
        trainer_pretraining.finetune_callbacks = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(cfg['data_dir'], cfg['pretraining_ckpt']),
            save_weights_only=True,
        )
        tf.random.set_seed(cfg['random_state'])
        trainer_pretraining.fit(dataset_is_train, dataset_is_val)

        trainer_args_pretraining = dict(
            {
                'pretraining_checkpoint': os.path.join(
                    cfg['data_dir'], cfg['pretraining_ckpt']
                )
            },
            **trainer_args
        )
        # Compute batch sizes
        num_max_batches = (
                (len(dataset_oos_trainval.keys) - (5 * 128 + 5 * 256 + 5 * 512)) // 1024
        )
        size_last_batch = (
                len(dataset_oos_trainval.keys) -
                (5 * 128 + 5 * 256 + 5 * 512 + num_max_batches * 1024)
        )
        batch_sizes: Final[List] = (
                5 * [128] + 5 * [256] + 5 * [512] + num_max_batches * [1024]
                + [size_last_batch]
        )

        for args, mode in zip(
                # [trainer_args_pretraining, trainer_args], ['warmstart', 'coldstart']
                [trainer_args], ['coldstart']
        ):

            args['num_workers'] = 1  # avoid file overload due to TF multi-processing
            trainer = WildlifeTrainer(**args)
            active_learner = ActiveLearner(
                trainer=trainer,
                pool_dataset=dataset_oos_trainval,
                label_file_path=os.path.join(cfg['data_dir'], cfg['label_file']),
                empty_class_id=empty_class_id,
                acquisitor_name='entropy',
                train_size=cfg['splits'][0],
                test_dataset=dataset_oos_test,
                test_logfile_path=os.path.join(
                    cfg['result_dir'], cfg['test_logfile'] + f'_{mode}.json'
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
                labels_supplied = [
                    (k, v) for k, v in label_dict.items() if k in keys_to_label
                ]
                save_as_csv(
                    labels_supplied,
                    os.path.join(cfg['active_dir'], 'active_labels.csv')
                )
                print('---> Supplied fresh labeled data')
                tf.random.set_seed(cfg['random_state'])
                active_learner.al_batch_size = batch_sizes[i + 1]
                active_learner.run()
                tf.keras.backend.clear_session()
                gc.collect()

            results = load_json(active_learner.test_logfile_path)
            results.update({'batch_sizes': batch_sizes})
            save_as_json(
                results,
                os.path.join(
                    cfg['result_dir'],
                    f'{TIMESTR}_results_oosample_active_{mode}.json'
                )
            )
    else:
        raise IOError('Unknown experiment')


if __name__ == '__main__':
    main()
