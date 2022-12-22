import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import cv2
import random
from typing import Dict, Final, List
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from wildlifeml.utils.io import (
    load_csv,
    load_json,
    load_pickle,
    save_as_csv,
    save_as_json,
    save_as_pickle,
)

from wildlifeml.utils.datasets import (
    map_bbox_to_img,
    map_preds_to_img,
    separate_empties,
)


def build_df_pred(
    eval_details_dict: dict,
    label_dict: dict,
    detector_dict: dict,
    image_data_dir: str,
) -> pd.DataFrame:

    keys_bbox = eval_details_dict['keys_bbox']
    preds_bbox = eval_details_dict['preds_bbox']
    preds_imgs = eval_details_dict['preds_imgs']
    img_name, true_class, pred_class, pred_score, n_preds = [], [], [], [], []
    pred_classes, pred_confs, md_confs, all_scores = [], [], [], []
    md_bboxs = []
    for key_img, pred_img in preds_imgs.items():
        ids_bbox = [i for i, key_bbox in enumerate(
            keys_bbox) if key_img in key_bbox]
        img_name.append(key_img)
        true_class.append(label_dict[key_img])
        pred_class.append(pred_img.argmax())
        pred_score.append(round(pred_img.max(), 3))
        n_preds.append(len(ids_bbox))
        pred_classes.append([x for x in preds_bbox[ids_bbox].argmax(1)])
        pred_confs.append([round(x, 3) for x in preds_bbox[ids_bbox].max(1)])
        md_confs.append(
            [
                round(detector_dict[k]['conf'], 2)
                for k in [keys_bbox[i] for i in ids_bbox]
            ]
        )
        md_bboxs.append(
            [
                detector_dict[k]['bbox']
                for k in [keys_bbox[i] for i in ids_bbox]
            ]
        )
        all_scores.append([round(x, 3) for x in pred_img])

    df = pd.DataFrame()
    df['img_name'] = img_name
    df['true_class'] = true_class
    df['pred_class'] = pred_class
    df['pred_score'] = pred_score
    df['n_preds'] = n_preds
    df['pred_classes'] = pred_classes
    df['pred_confs'] = pred_confs
    df['md_confs'] = md_confs
    df['md_bboxs'] = md_bboxs
    df['all_scores'] = all_scores
    df['img_path'] = [os.path.join(image_data_dir, name) for name in img_name]
    df['true_class'] = df['true_class'].astype(float).astype(int)
    df['pred_class'] = df['pred_class'].astype(float).astype(int)
    return df


def labelize_df_pred(
    df_pred: pd.DataFrame,
    label_map: dict,
) -> pd.DataFrame:

    inverse_map = {v: k for k, v in label_map.items()}
    df = df_pred.copy()
    df['true_class'] = df['true_class'].map(inverse_map)
    df['pred_class'] = df['pred_class'].map(inverse_map)
    df['pred_classes'] = [
        [inverse_map[p] for p in ps] for ps in df['pred_classes']
        ]
    return df


def get_labels(dataset, label_dict):
    bbox_keys = dataset.keys
    img_keys = [map_bbox_to_img(k) for k in bbox_keys]
    img_labels = [label_dict[k] for k in img_keys]
    return img_labels


def inspect_confusion(df_pred, normalize=True, labels=None, ax=None):
    y_true = df_pred['true_class']
    y_pred = df_pred['pred_class']

    if labels is None:
        labels = np.unique(y_true).tolist()
    n_classes = len(labels)

    cm_norm = confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        normalize='true',
    )
    if normalize:
        cm = cm_norm
    else:
        cm = confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            normalize=None,
            )

    sns.set_theme(style='white')
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(n_classes), labels, fontsize=9, rotation=30)
    ax.set_yticks(np.arange(n_classes), labels, fontsize=9)
    for i, j in itertools.product(range(n_classes), range(n_classes)):
        ax.text(
            j, i,
            format(cm_norm[i, j], '.2f') if normalize else format(cm[i, j]),
            fontsize=10, horizontalalignment="center",
            color="white" if cm_norm[i, j] > cm_norm.max()/2. else "black",
        )
    if ax is None:
        plt.show()
        plt.close()


def inspect_results(
    df_pred: pd.DataFrame,
    test_label: str,
    label_map: dict,
    n_displays: int,
    is_truth: bool,
    sorting='descending',
    which_preds='only_false',
):
    # sorting can be: ['descending', 'ascending', 'random']
    # which_preds can be: ['only_true', 'only_false', 'mixed', 'IMAGE_NAME.JPG']

    if n_displays == 0:
        print('Please increase the number of displaying images (n_displays).')
        return
    specific_image = False
    if which_preds == 'only_true':
        df = df_pred[df_pred['true_class'] == df_pred['pred_class']]
    elif which_preds == 'only_false':
        df = df_pred[df_pred['true_class'] != df_pred['pred_class']]
    elif which_preds == 'mixed':
        df = df_pred
    else:
        specific_image = True
        df = df_pred[df_pred['img_name'] == which_preds]
    if not specific_image:
        if is_truth:
            df = df[df['true_class'] == test_label]
        else:
            df = df[df['pred_class'] == test_label]
    df = df.sort_values(
        by=['pred_score'], ascending=False, ignore_index=True,
    )

    n_available = len(df)
    if n_displays > n_available:
        print(f'There are {n_available} available images to be displayed.')
    n_displays = min(n_displays, n_available)
    all_ids = list(df.index)
    if sorting == 'ascending':
        ids = all_ids[:n_displays]
    elif sorting == 'descending':
        ids = all_ids[:-n_displays-1:-1]
    else:
        ids = random.choices(all_ids, k=n_displays)

    for index in ids:
        score_dict = dict(
            zip(label_map.keys(), df.loc[index, 'all_scores'])
        )
        score_dict = dict(
            sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        )
        img_path = df.loc[index, 'img_path']
        md_confs = df.loc[index, 'md_confs']
        md_bboxs = df.loc[index, 'md_bboxs']
        n_preds = df.loc[index, 'n_preds']

        fig, ax = plt.subplots(figsize=(6, 6))
        image = cv2.imread(img_path)
        for index_bbox in range(n_preds):
            md_bbox = md_bboxs[index_bbox]
            md_conf = md_confs[index_bbox]
            image = draw_bbox(bbox=md_bbox, image=image, text=str(md_conf))
        ax.imshow(image)
        plt.show()
        plt.close()
        print(f"img_name: {df.loc[index, 'img_name']}")
        print(f"true_class: {df.loc[index, 'true_class']}")
        print(f"pred_class: {df.loc[index, 'pred_class']}")
        print(f"pred_score: {df.loc[index, 'pred_score']}")
        print(f"n_preds: {df.loc[index, 'n_preds']}")
        print(f"pred_classes: {df.loc[index, 'pred_classes']}")
        print(f"pred_confs: {df.loc[index, 'pred_confs']}")
        print(f"md_confs: {df.loc[index, 'md_confs']}")
        print(f"score_dict: {score_dict}")


def plot_frequencies(df, label_map, ax=None):
    sns.set_style("darkgrid")
    frequencies = get_frequencies(df, label_map, normalize=True)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(x=frequencies)
    if ax is None:
        plt.show()
        plt.close()


def plot_class_performance(df_pred, df_meta, label_map, ax=None):

    labels = list(label_map.keys())
    n_classes = len(labels)
    report = classification_report(
        y_true=df_pred['true_class'],
        y_pred=df_pred['pred_class'],
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    frequencies = get_frequencies(df_meta, label_map, normalize=True)
    precisions = [report[label]['precision'] for label in labels]
    recalls = [report[label]['recall'] for label in labels]
    f1s = [report[label]['f1-score'] for label in labels]
    x = np.arange(n_classes)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.1
    ax.bar(
        x=x + 0 * width,
        height=frequencies,
        width=width,
        color='black',
        label='frequency',
    )
    ax.bar(
        x=x + 1 * width,
        height=precisions,
        width=width,
        color='#165C8D',
        label='precision',
    )
    ax.bar(
        x=x + 2 * width,
        height=recalls,
        width=width,
        color='#167F0F',
        label='recall',
    )
    ax.bar(
        x=x + 3 * width,
        height=f1s,
        width=width,
        color='#CE0917',
        label='f1',
    )
    ax.set_xticks(x + width, labels, rotation=30, fontsize=8)
    ax.legend(loc='best', fontsize=7)
    if ax is not None:
        plt.show()
        plt.close()


def get_frequencies(df, label_map, normalize=True):
    frequencies = [
        Counter(df['true_class'])[label]for label in label_map.keys()
    ]
    if normalize:
        frequencies = [freq/np.sum(frequencies) for freq in frequencies]
    return np.array(frequencies)


def convert_bbox(bbox, image_shape):
    sx = image_shape[1]
    sy = image_shape[0]
    x1, y1, wx, wy = bbox
    x2 = x1 + wx
    y2 = y1 + wy
    x1 *= sx
    x2 *= sx
    y1 *= sy
    y2 *= sy
    start_point = int(x1), int(y1)
    end_point = int(x2), int(y2)
    return start_point, end_point


def draw_bbox(bbox, image, text):

    bbox_color = (255, 0, 0)
    text_color = (0, 255, 0)
    bbox_thickness = 2
    text_thickness = 2
    text_scale = .8
    start_point, end_point = convert_bbox(bbox, image.shape)

    image = cv2.rectangle(
        image,
        start_point,
        end_point,
        bbox_color,
        bbox_thickness,
    )
    if text is not None:
        image = cv2.putText(
            image,
            text,
            start_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            text_color,
            text_thickness,
            cv2.LINE_AA,
        )
    return image


def evaluate_performance(y_true, y_pred, labels, average='macro', pos_label=1):
    if labels is None:
        labels = np.unique(y_true).tolist()

    pref_dict = {
        'acc': accuracy_score(
            y_true=y_true,
            y_pred=y_pred,
            ),
        'prec': precision_score(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            average=average,
            zero_division=0,
            pos_label=pos_label,
            ),
        'rec': recall_score(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            average=average,
            zero_division=0,
            pos_label=pos_label,
            ),
        'f1': f1_score(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            average=average,
            zero_division=0,
            pos_label=pos_label,
        ),
        'cm': confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            normalize=None,
            ),
        'cm_norm': confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            normalize='true',
            ),
    }
    return pref_dict


def get_binary_confusion_md(dataset, threshold, repo_dir):

    cfg: Final[Dict] = load_json(os.path.join(repo_dir, 'configs/cfg.json'))

    label_dict = {
        k: v
        for k, v in load_csv(os.path.join(cfg['data_dir'], cfg['label_file']))
    }

    empty_class_id = load_json(
        os.path.join(cfg['data_dir'], 'label_map.json')
    ).get('empty')

    true_empty = set(
        [k for k, v in label_dict.items() if v == str(empty_class_id)]
    )
    true_nonempty = set(label_dict.keys()) - set(true_empty)

    # Get imgs that MD classifies as empty
    keys_empty_bbox, keys_nonempty_bbox = separate_empties(
        os.path.join(cfg['data_dir'], cfg['detector_file']), float(threshold)
    )
    keys_empty_bbox = list(
        set(keys_empty_bbox).intersection(set(dataset.keys))
    )
    keys_nonempty_bbox = list(
        set(keys_nonempty_bbox).intersection(set(dataset.keys))
    )
    keys_empty_img = list(
        set([map_bbox_to_img(k) for k in keys_empty_bbox])
    )
    keys_nonempty_img = list(
        set([map_bbox_to_img(k) for k in keys_nonempty_bbox])
    )
    # Compute confusion metrics for MD stand-alone
    tn = len(true_empty.intersection(set(keys_empty_img)))
    tp = len(true_nonempty.intersection(set(keys_nonempty_img)))
    fn = len(true_nonempty.intersection(set(keys_empty_img)))
    fp = len(true_empty.intersection(set(keys_nonempty_img)))

    conf_md = {
        'tnr': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'fnr': fn / (tp + fn) if (tp + fn) > 0 else 0.0,
        'fpr': fp / (tn + fp) if (tn + fp) > 0 else 0.0,
    }
    return conf_md


def get_binary_confusion_ppl(y_true, y_pred):
    tp, tn, fp, fn = 0, 0, 0, 0
    for true, pred in zip(y_true, y_pred):
        if true == 'empty':
            if true == pred:
                tn += 1
            else:
                fp += 1
        else:
            if pred == 'empty':
                fn += 1
            else:
                tp += 1
    conf_ppl = {
        'tnr': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'fnr': fn / (tp + fn) if (tp + fn) > 0 else 0.0,
        'fpr': fp / (tn + fp) if (tn + fp) > 0 else 0.0,
    }
    return conf_ppl


def _escape_latex(s):
    return (
        s.replace("\\", "ab2§=§8yz")  # rare string for final conversion
        .replace("ab2§=§8yz ", "ab2§=§8yz\\space ")  # backslash gobbles spaces
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~ ", "~\\space ")  # since \textasciitilde gobbles spaces
        .replace("~", "\\textasciitilde ")
        .replace("^ ", "^\\space ")  # since \textasciicircum gobbles spaces
        .replace("^", "\\textasciicircum ")
        .replace("ab2§=§8yz", "\\textbackslash ")
    )


def prepare_latex(df):
    old = [c for c in df.columns]
    new = [_escape_latex(c) for c in df.columns]
    df_tex = df.rename(columns=dict(zip(old, new)))
    return df_tex
