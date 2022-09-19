# wildlife-experiments

## Requirements

Either install with `requirements.txt` or `environment.yml` (e.g., via `conda`).

## Source repo
Currently living on GitHub.

Install via ``pip install git+https://<PAT>:@github.com/slds-lmu/wildlife-ml.git``, 
using personal access token (the PAT might not be necessary if working on a private 
machine).

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
- **Note**: You should re-run `run_prep.sh` before executing new experiments (for example, active learning modifies some used files).