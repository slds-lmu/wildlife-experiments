"""In-sample results."""

import click
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

N_GPU = len(os.environ['CUDA_VISIBLE_DEVICES'])
N_CPU: Final[int] = 16
THRESH_PROGRESSIVE: Final[float] = 0.5
THRESH_NOROUZZADEH: Final[float] = 0.9


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
        k: v for k, v in load_csv(os.path.join(cfg['data_dir'], cfg['label_file']))
    }
    stations_dict = {
        k: {'station': v}
        for k, v in load_csv(os.path.join(cfg['data_dir'], cfg['meta_file']))
    }

    # Get truly empty images

    empty_class = load_json(
        os.path.join(cfg['data_dir'], 'label_map.json')
    ).get('empty')
    true_empty = set([k for k, v in label_dict.items() if v == str(empty_class)])
    true_nonempty = set(label_dict.keys()) - set(true_empty)

    # Prepare training

    dataset_is_train = load_pickle(
        os.path.join(cfg['data_dir'], 'dataset_is_train.pkl')
    )
    dataset_is_val = load_pickle(os.path.join(cfg['data_dir'], 'dataset_is_val.pkl'))
    dataset_is_trainval = load_pickle(os.path.join(
        cfg['data_dir'], 'dataset_is_trainval.pkl')
    )
    dataset_is_test = load_pickle(os.path.join(cfg['data_dir'], 'dataset_is_test.pkl'))
    dataset_oos_train = load_pickle(
        os.path.join(cfg['data_dir'], 'dataset_oos_train.pkl')
    )
    dataset_oos_test = load_pickle(os.path.join(
        cfg['data_dir'], 'dataset_oos_test.pkl')
    )

    trainer = WildlifeTuningTrainer(
        search_space={
            'backbone': ray.tune.choice(['resnet50']),
            'transfer_learning_rate': ray.tune.choice([1e-4]),
            'finetune_learning_rate': ray.tune.choice([1e-4]),
            'batch_size': ray.tune.choice([32])
        },
        loss_func=keras.losses.SparseCategoricalCrossentropy(),
        num_classes=cfg['num_classes'],
        transfer_epochs=cfg['transfer_epochs'],
        finetune_epochs=cfg['finetune_epochs'],
        transfer_optimizer=Adam(),
        finetune_optimizer=Adam(),
        finetune_layers=cfg['finetune_layers'],
        transfer_callbacks=None,
        finetune_callbacks=None,
        num_workers=cfg['num_workers'],
        eval_metrics=cfg['eval_metrics'],
        resources_per_trial={'cpu': 4, 'gpu': N_GPU},
        max_concurrent_trials=1,
        time_budget=10,
        # search_alg_id='randomsearch',
        # scheduler_alg_id='fifoscheduler',
        # objective='accuracy',
    )

    evaluator_is = Evaluator(
        label_file_path=os.path.join(cfg['data_dir'], cfg['label_file']),
        detector_file_path=os.path.join(cfg['data_dir'], cfg['detector_file']),
        dataset=dataset_is_test,
        num_classes=trainer.get_num_classes(),
    )
    evaluator_oos = Evaluator(
        label_file_path=os.path.join(cfg['data_dir'], cfg['label_file']),
        detector_file_path=os.path.join(cfg['data_dir'], cfg['detector_file']),
        dataset=dataset_oos_test,
        num_classes=trainer.get_num_classes(),
    )

    # ----------------------------------------------------------------------------------
    # IN-SAMPLE ------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    # EMPTY VS NON-EMPTY ---------------------------------------------------------------

    if experiment == 'insample_empty':

        # Compute empty-detection performance of MD stand-alone and for entire pipeline

        results_empty = {
            'names': ['ours', 'progressive', 'norouzzadeh'],
            'thresholds': [cfg['md_conf'], THRESH_PROGRESSIVE, THRESH_NOROUZZADEH]
        }
        tnr_md, tpr_md, fnr_md, fpr_md = [], [], [], []
        tnr_ppl, tpr_ppl, fnr_ppl, fpr_ppl = [], [], [], []

        for threshold in results_empty['thresholds']:

            # Get imgs that MD classifies as empty
            keys_empty_bbox, keys_nonempty_bbox = separate_empties(
                os.path.join(cfg['data_dir'], cfg['detector_file']), threshold
            )
            keys_empty_bbox = list(
                set(keys_empty_bbox).intersection(set(dataset_is_trainval.keys))
            )
            keys_nonempty_bbox = list(
                set(keys_nonempty_bbox).intersection(set(dataset_is_trainval.keys))
            )
            keys_empty_img = list(set([map_bbox_to_img(k) for k in keys_empty_bbox]))
            keys_nonempty_img = list(
                set([map_bbox_to_img(k) for k in keys_nonempty_bbox])
            )

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

            trainer_empty = deepcopy(trainer)
            print('---> Training on wildlife data')
            trainer_empty.fit(
                train_dataset=dataset_train_thresh, val_dataset=dataset_val_thresh
            )
            print('---> Evaluating on test data')
            evaluator = Evaluator(
                label_file_path=os.path.join(cfg['data_dir'], cfg['label_file']),
                detector_file_path=os.path.join(cfg['data_dir'], cfg['detector_file']),
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

        save_as_json(
            results_empty,
            os.path.join(cfg['result_dir'], 'results_insample_empty.json')
        )

    # PERFORMANCE ----------------------------------------------------------------------

    elif experiment == 'insample_perf':

        trainer_perf_is = deepcopy(trainer)

        print('---> Training on wildlife data')
        trainer_perf_is.fit(train_dataset=dataset_is_train, val_dataset=dataset_is_val)
        print('---> Evaluating on test data')
        results_perf = evaluator_is.evaluate(trainer_perf_is)

        save_as_json(
            results_perf, os.path.join(cfg['result_dir'], 'results_insample_perf.json')
        )

    # BENEFIT OF TUNING ----------------------------------------------------------------

    elif experiment == 'insample_tuning':

        # TODO: set hyperparams appropriately
        trainer_untuned_default = WildlifeTrainer(
            loss_func=keras.losses.SparseCategoricalCrossentropy(),
            num_classes=cfg['num_classes'],
            transfer_epochs=cfg['transfer_epochs'],
            finetune_epochs=cfg['finetune_epochs'],
            transfer_optimizer=Adam(),
            finetune_optimizer=Adam(),
            finetune_layers=cfg['finetune_layers'],
            transfer_callbacks=None,
            finetune_callbacks=None,
            num_workers=cfg['num_workers'],
            eval_metrics=cfg['eval_metrics'],
        )
        trainer_untuned_random = WildlifeTrainer(
            loss_func=keras.losses.SparseCategoricalCrossentropy(),
            num_classes=cfg['num_classes'],
            transfer_epochs=cfg['transfer_epochs'],
            finetune_epochs=cfg['finetune_epochs'],
            transfer_optimizer=Adam(),
            finetune_optimizer=Adam(),
            finetune_layers=cfg['finetune_layers'],
            transfer_callbacks=None,
            finetune_callbacks=None,
            num_workers=cfg['num_workers'],
            eval_metrics=cfg['eval_metrics'],
        )

        print('---> Training on wildlife data')
        # TODO think about necessity of val data here (tuning trainer trained on both)
        trainer_untuned_default.fit(
            train_dataset=dataset_is_train, val_dataset=dataset_is_val
        )
        trainer_untuned_random.fit(
            train_dataset=dataset_is_train, val_dataset=dataset_is_val
        )

        print('---> Evaluating on test data')
        results_tuning = {
            'untuned_default': evaluator_is.evaluate(trainer_untuned_default),
            'untuned_random': evaluator_is.evaluate(trainer_untuned_random),
        }

        save_as_json(
            results_tuning,
            os.path.join(cfg['result_dir'], 'results_insample_tuning.json')
        )

    # ----------------------------------------------------------------------------------
    # OUT-OF-SAMPLE --------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    # WITHOUT AL -----------------------------------------------------------------------

    elif experiment == 'oosample_perf':

        print('---> Training on in-sample data')
        trainer_perf_oos = deepcopy(trainer)
        trainer_perf_oos.fit(train_dataset=dataset_is_train, val_dataset=dataset_is_val)

        print('---> Evaluating on out-of-sample data')
        results_perf_passive = evaluator_oos.evaluate(trainer_perf_oos)

        save_as_json(
            results_perf_passive,
            os.path.join(cfg['result_dir'], 'results_oosample_perf.json')
        )

    # WITH AL (WARM- AND COLDSTART) ----------------------------------------------------

    elif experiment == 'oosample_active':

        # trainer_pretraining = WildlifeTrainer(
        #     loss_func=keras.losses.SparseCategoricalCrossentropy(),
        #     batch_size=cfg['batch_size'],
        #     num_classes=cfg['num_classes'],
        #     transfer_epochs=1,
        #     finetune_epochs=1,
        #     transfer_optimizer=Adam(),
        #     finetune_optimizer=Adam(),
        #     finetune_layers=cfg['finetune_layers'],
        #     transfer_callbacks=None,
        #     finetune_callbacks=keras.callbacks.ModelCheckpoint(
        #         filepath=os.path.join(cfg['data_dir'], cfg['pretraining_ckpt']),
        #         save_weights_only=True,
        #     ),
        #     num_workers=cfg['num_workers'],
        #     eval_metrics=cfg['eval_metrics'],
        # )
        # trainer_pretraining.fit(dataset_is_trainval, dataset_is_test)

        trainer_active = WildlifeTrainer(
            loss_func=keras.losses.SparseCategoricalCrossentropy(),
            batch_size=cfg['batch_size'],
            num_classes=cfg['num_classes'],
            transfer_epochs=cfg['transfer_epochs'],
            finetune_epochs=cfg['finetune_epochs'],
            transfer_optimizer=Adam(),
            finetune_optimizer=Adam(),
            finetune_layers=cfg['finetune_layers'],
            transfer_callbacks=None,
            finetune_callbacks=None,
            num_workers=cfg['num_workers'],
            eval_metrics=cfg['eval_metrics'],
            pretraining_checkpoint=os.path.join(
                cfg['data_dir'], cfg['pretraining_ckpt']
            )
        )

        for trainer_obj, mode in zip(
                [trainer_active, deepcopy(trainer)], ['warmstart', 'coldstart']
        ):

            learner = ActiveLearner(
                trainer=trainer_obj,
                pool_dataset=dataset_oos_train,
                label_file_path=os.path.join(cfg['data_dir'], cfg['label_file']),
                empty_class_id=load_json(os.path.join(
                    cfg['data_dir'], 'label_map.json')).get('empty'),
                acquisitor_name='entropy',
                train_size=cfg['splits'][0],
                test_dataset=dataset_oos_test,
                test_logfile_path=os.path.join(
                    cfg['result_dir'], cfg['test_logfile'] + f'{mode}.json'
                ),
                meta_dict=stations_dict,
                active_directory=cfg['active_dir'],
                state_cache=os.path.join(cfg['active_dir'], '.activecache.json')
            )

            print('---> Running initial AL iteration')
            if os.path.exists(os.path.join(cfg['active_dir'], '.activecache.json')):
                os.remove(os.path.join(cfg['active_dir'], '.activecache.json'))
            learner.run()
            learner.do_fresh_start = False

            for i in range(cfg['al_iterations']):
                print(f'---> Starting AL iteration {i + 1}')
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
                learner.run()

            results = load_json(learner.test_logfile_path)
            save_as_json(
                results,
                os.path.join(cfg['result_dir'], f'results_oosample_active_{mode}.json')
            )

    else:
        raise IOError('Unknown experiment')


if __name__ == '__main__':
    main()