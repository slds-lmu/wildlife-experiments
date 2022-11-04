# wildlife-experiments

## Installation

- Clone the current repo into your local machine: `git clone https://github.com/slds-lmu/wildlife-experiments.git`
- Install the required dependencies either with `requirements.txt` or `environment.yml`: `conda env create -f environment.yml`
  - :bulb: `environment.yml` creates a conda env and installs the dependencies on it. 
  - :bulb: The created env is called `wex`, but, you can customize its name by editing the first line of `environment.yml`.
- Install the `wildlife-ml` package on the `wex`: `pip install git+https://<PAT>:@github.com/slds-lmu/wildlife-ml.git`
  - :bulb: The PAT (i.e., personal access token) might not be necessary if working on a private machine. 
  - :bulb: If you wish to install `wildlife-ml` from a branch other than `main` then use: `pip install git+https://<PAT>:@github.com/slds-lmu/wildlife-ml.git@<branch-name>`


If you wish to install from a branch other thain `main`, just add `@<branch-name>` to the above URL (at the end).

## Execution

- All relevant files can be found under `scripts`.
- You need to run the following files with arguments:
  - `bash run_prep.sh <GPU>`
    - If you wish to use a GPU, enter here the number(s) of the GPUs that shall be visible to CUDA.
    - There are three optional file path arguments you can hand over in the specified order to override the default.
  - `bash run_experiments.sh <GPU> <experiment>`
    - In addition to the GPU, specify the experiment you wish to run.
    - Again, there is an optional file path argument.
    - E.g., `bash run_experiments.sh 1 insample_perf`
:bulb: Re-run `run_prep.sh` before executing new experiments (for example, active learning modifies some used files).
