# wildlife-experiments

This repository implements experiments and exemplary training scripts using the functionalities of the source repository `wildlife-ml`, which, in turn, contains code for classifying wildlife images with an active learning option.

## Disclaimer

- The `main` branch is protected to avoid unintended modifications.
- Please **always** work on a local branch -- also if you only run the code, as the produced files will otherwise overwrite the original, which may cause conflicts and result in lost work.

## Installation

- **This repository**: clone to your local machine via: `git clone git@github.com:slds-lmu/wildlife-experiments.git`.
- **Virtual environment**: this might be a little iffy due to the use of `tensorflow`, depending on the settings of your machine (especially if you plan to work with GPUs). We provide a environment that works for `CUDA` in version 11.1, but you might need to use different versions of `tensorflow` / `tensorflow-gpu`.
  - :bulb: For example, using `conda`, run `conda env create -f environment.yml`.
  - :bulb: `environment.yml` creates a `conda` environment and installs the dependencies on it. 
  - :bulb: The created environment is called `wildlife`, but you can customize its name by editing the first line of `environment.yml`.
- **Source package**: the above environment includes packages available via `pip` / `conda`, but as our `wildlifeml` package is only available via GitHub, you need to install it separately via `pip install git+https://<PAT>:@github.com/slds-lmu/wildlife-ml.git` (within the virtual environment, which, on `conda`, is fired up via `conda activate <environment name>`).
  - :bulb: The cloning statement contains a PAT (personal access token), which might not be necessary if working on a private machine. 
  - :bulb: If you wish to install `wildlifeml` from a branch other than `main`, then use `pip install git+https://<PAT>:@github.com/slds-lmu/wildlife-ml.git@<branch-name>`, but note that this is usually work in progress that might not have been tested yet.

## Execution

- All relevant files can be found under `scripts`.
- For **EXEMPLARY USE**:
  - `compute_example.py` contains examples of how to use this repository.
  - It provides code for 3 tasks: preparing the data ("prep") and training in standard ("train_passive") as well as active ("train_active") fashion.
  - In order to execute the code, run the corresponding bash file (the sole purpose of which is to execute the code in `compute_example.py` with appropriate settings) from your command line via `bash scripts/run_example.sh <path/to/config/file> <task>`, e.g., `bash scripts/run_example.sh /home/me/wildlife-experiments/configs/config_example.json prep`.
  - There are two places that require custom user input:
    - `run_example.sh` expects two arguments: the (absolute) path to the configuration file that contains all training specifications (see next step), and the task to be executed (must be one out of "prep", "train_passive", "train_active").
    - `configs/config_example.json` stores all customizable options for the exemplary code. Find examples for the required files in the folder `example_files`.
      - `label_file_train`: path to csv file with training data labels. Must consist of two columns, the first containing image names, the second image labels. Please note that the class of empty images must be named "empty".
      - `label_file_test`: path to csv file like `label_file_train`.
      - `label_file_pretrain`: (OPTIONAL) path to csv file like `label_file_train`. If no file shall be provided, specify the path as "".
      - `label_file_prod`: (OPTIONAL) path to csv file like `label_file_train`. If no file shall be provided, specify the path as "".
      - `detector_file_train`: path to json file where MegaDetector results shall be stored.
      - `detector_file_test`: path to json file where MegaDetector results shall be stored.
      - `detector_file_pretrain`: (OPTIONAL) path to json file where MegaDetector results shall be stored.
      - `detector_file_prod`: (OPTIONAL) path to json file where MegaDetector results shall be stored.
      - `meta_file_train`: (OPTIONAL) csv file with additional meta data to use for stratified splitting. Must consist of two columns, the first containing image names, the second a stratification variable (e.g., camera stations).
      - `data_dir`: path to directory where intermediate data-related files shall be stored (directory must exist).
      - `img_dir`: path to directory where images are stored.
      - `md_batchsize`: batch size for MegaDetector. Typically, 2 to the power of something to ensure adequate use of computational resources.
      - `md_conf`: confidence threshold for MegaDetector (i.e., which confidence level should the MegaDetector have in order for an image to be deemed non-empty?). Images with bounding boxes below this threshold are considered empty and excluded from further training (but included in the evaluation).
      - `batch_size`: batch size for training. Typically, 2 to the power of something to ensure adequate use of computational resources.
      - `splits`: split ratio for train / validation split of training data. The original implementation supports a three-way split, but here only a two-way split is used. The first number refers to the train ratio, the third to the validation ratio, the second should be set to 0.
      - `random_state`: integer number to be used as random seed.
      - `num_classes`: number of classes in the data (incl. empty class).
      - `transfer_epochs`: number of epochs for transfer learning (i.e., training the linear classification head placed on top of the pre-trained feature extractor).
      - `finetune_epochs`: number of epochs for fine-tuning (i.e., re-training (parts of the) pre-trained feature extractor).
      - `transfer_learning_rate`: learning rate for transfer learning.
      - `finetune_learning_rate`: learning rate for fine-tuning.
      - `finetune_layers`: number of layers of the feature extractor that shall be re-trained (e.g., 1 means the last layer before the linear classification head is re-trained).
      - `model_backbone`: pre-trained backbone to be used. Must be one of: "resnet50", "inceptionresnetv2", "vgg19", "xception", "densenet121", "densenet201".
      - `num_workers`: number of workers available for training (depends on your machine, see, e.g., https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras).
      - `result_file`: path to json file in which results shall be stored.
      - `label_file_active`: path to csv file where labels acquired during active learning shall be stored. Please do not name this file `active_labels.csv`, as this name is used during the annotation step in the active learning loop and refers to a file that is regularly overwritten.
      - `acquisition_function`: acquisition criterion used in active learning. Must be one of: "random" (random sampling), "entropy" (highest softmax entropy as a proxy of predictive uncertainty), "breakingties" (smallest gap between first- and second-largest class probability as a proxy of predictive uncertainty).
      - `active_dir`: path to directory where images and files for the annotation step in the active learning loop shall be stored.
      - `al_iterations`: number of active learning iterations.
      - `al_batch_size`: number of images to be selected for annotation in each active learning iteration.
      - `human_annotation`: will there be an actual human labeling images during active learning? If False, labels will be iteratively fetched from `label_file_train`, simulating active learning.
- Only for the **PAPER EXPERIMENTS**:
  - You need to run the following files (in this order) with arguments:
    - `bash scripts/run_prep.sh <GPU> <personal folder>`
      - Creates necessary intermediate files, possibly runs the MegaDetector, and creates different data partitions (in-sample, out-of-sample, train/val/test splits etc.).
      - If you wish to use a GPU, enter here the number(s) of the GPUs that shall be visible to CUDA.
      - Specify the personal folder in which the `wildlife-experiments` repository is stored (no trailing slash).
      - There are three optional file path arguments you can hand over in the specified order to override the default.
    - `bash run_experiments.sh <GPU> <personal folder> <experiment>`
      - Performs different experiments as chosen to provide evidence for the claims of the paper.
      - In addition to the GPU and personal folder, specify the experiment you wish to run.
      - Again, there is an optional file path argument.
      - E.g., `bash scripts/run_experiments.sh 1 /home/wimmerl/projects insample_perf`
:bulb: Re-run `scripts/run_prep.sh` before executing new experiments (for example, active learning modifies some used files).