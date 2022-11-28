import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import itertools
from wildlifeml.utils.datasets import map_bbox_to_img
import cv2
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    classification_report,
    confusion_matrix)


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
        pred_classes.append([format(x)for x in preds_bbox[ids_bbox].argmax(1)])
        pred_confs.append([round(x, 3) for x in preds_bbox[ids_bbox].max(1)])
        md_confs.append(
            [
                round(detector_dict[k]['conf'], 2)
                for k in [keys_bbox[i] for i in ids_bbox]
            ]
        )
        md_bboxs.append(
            [detector_dict[k]['bbox'] for k in [keys_bbox[i] for i in ids_bbox]]
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
    df['true_class'] = df['true_class'].astype('str')
    df['pred_class'] = df['pred_class'].astype('str')
    return df


def labelize_df_pred(
    df_pred: pd.DataFrame,
    translator_dict: dict,
) -> pd.DataFrame:

    df = df_pred.copy()
    df['true_class'] = df['true_class'].map(translator_dict)
    df['pred_class'] = df['pred_class'].map(translator_dict)
    df['pred_classes'] = [[translator_dict[p] for p in ps] for ps in df['pred_classes']]
    return df


def display_image(img_path, figsize=(6, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(mpimg.imread(img_path))
    plt.show()
    plt.close()


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


def inspect_misclasses(
    df_pred: pd.DataFrame,
    test_label: str,
    translator_dict: dict,
    n_displays: int,
    descending: bool,
    is_truth: bool,
):

    df_miss = df_pred[df_pred['true_class'] != df_pred['pred_class']]

    if n_displays == 0:
        print('No misclassification for the specified class :-)')
        return
    else:
        if is_truth:
            df_miss = df_miss[df_miss['true_class'] == test_label]
        else:
            df_miss = df_miss[df_miss['pred_class'] == test_label]
        df_miss = df_miss.reset_index(drop=True)
        n_displays = min(n_displays, len(df_miss))

        for index in range(n_displays):
            score_dict = dict(
                zip(translator_dict.values(), df_miss.loc[index, 'all_scores'])
            )
            score_dict = dict(
                sorted(score_dict.items(), key=lambda x: x[1], reverse=descending)
            )
            img_path = df_miss.loc[index, 'img_path']
            md_confs = df_miss.loc[index, 'md_confs']
            md_bboxs = df_miss.loc[index, 'md_bboxs']
            n_preds = df_miss.loc[index, 'n_preds']

            fig, ax = plt.subplots(figsize=(6, 6))
            image = cv2.imread(img_path)
            for index_bbox in range(n_preds):
                md_bbox = md_bboxs[index_bbox]
                md_conf = md_confs[index_bbox]
                image = draw_bbox(bbox=md_bbox, image=image, text=str(md_conf))
            ax.imshow(image)
            plt.show()
            plt.close()
            print(f"img_name: {df_miss.loc[index, 'img_name']}")
            print(f"true_class: {df_miss.loc[index, 'true_class']}")
            print(f"pred_class: {df_miss.loc[index, 'pred_class']}")
            print(f"pred_score: {df_miss.loc[index, 'pred_score']}")
            print(f"n_preds: {df_miss.loc[index, 'n_preds']}")
            print(f"pred_classes: {df_miss.loc[index, 'pred_classes']}")
            print(f"pred_confs: {df_miss.loc[index, 'pred_confs']}")
            print(f"md_confs: {df_miss.loc[index, 'md_confs']}")
            print(f"score_dict: {score_dict}")


def plot_frequencies(df, labels, ax=None):
    sns.set_style("darkgrid")
    frequencies = get_frequencies(df, labels, normalize=True)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(x=frequencies)
    if ax is None:
        plt.show()
        plt.close()


def plot_class_performance(df_pred, df_meta, labels, ax=None):

    n_classes = len(labels)
    report = classification_report(
        y_true=df_pred['true_class'],
        y_pred=df_pred['pred_class'],
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    frequencies = get_frequencies(df_meta, labels, normalize=True)
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
        x= x + 1 * width,
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


def get_frequencies(df, labels, normalize=True):
    frequencies = [Counter(df['true_class'])[label] for label in labels]
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
