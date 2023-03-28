# wildlife-experiments

This repository contains a variety of experiments implementing the analytic pipeline for wildlife image classification designed in [`wildlife-ml`](https://github.com/slds-lmu/wildlife-ml).
Our approach is two-fold, combining improved strategies for object detection and image classification with an active learning system that allows for more efficient training of deep learning models. 
Please also see [https://github.com/slds-lmu/wildlife-ml](https://github.com/slds-lmu/wildlife-ml) for a detailed description.

## Installation

We provide our full Python environment in `env.yml`.

## Workflow

### 01: Folder structure

```
├── configs
    ├── cfg.son
├── data
    ├── images_megadetector.json
    ├── ...
    └── labels.csv
├── results
└── scripts
    ├── execute_experiments.py
    ├── ...
    └── run_experiments.sh

### 02: Order of execution

- The scripts are executed in the following order:
  - `prep_config.py`
  - `prep_data.py`
  - Optional: `execute_tuning.py`
  - `execute_expeiments`
- Each Python script has an accompanying bash file that allows for convenient, step-by-step command-line execution with minimal user input.
- `prep_config.py` specifies user preferences, in particular:
  - `img_dir`: path to directory where images are stored. Please note: all images should be in a single directory, allocation to different datasets is done via separate files containing the respectively relevant keys. 
  - `data_dir`: path to directory where intermediate data-related files shall be stored (directory must exist).
  - `result_dir`: path to directory where results shall be stored (directory must exist).
  - `active_dir`: path to directory where data related to active learning (e.g., images to label in current iteration) shall be stored.
- Further settings in `prep_config.py` may be adapted if desired.

## Coming soon 

We are currently working on deploying a server to host data and the active-learning system.
