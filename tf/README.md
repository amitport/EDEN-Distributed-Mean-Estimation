# EDEN evaluation: federated learning on EMNIST and Shakespeare
## Setup

### Install requirements:

```setup
pip install -r requirements.txt
```

### Initialize git submodule

Run the following to make sure that the remote Google's [federated research repo](https://github.com/google-research/federated) is cloned as a submodule:

```setup
git submodule update --init --recursive
```

### Update PYTHONPATH

Add `tf/google_research_federated` to `PYTHONPATH`.

## Training

In order to reproduce the paper's results, execute `trainer.py` (the current working directory should be the repo's root).

You can view the documentation for every command line parameter using `trainer.py --help`.

* `--root_output_dir` and `--experiment_name` flags determine where the outputs will be stored. 

* `--compressor` can be one of `eden`, `kashin`, `hadamard`, or `sq`, as described in the paper.

* `--num-bits` determines the number of integer bits to use and `--p` is the sparsification factor. For example for 4 bits: `--num-bits=4 --p=1`; and for 0.1 bit `--num-bits=1 --p=0.1`.

* The rest of the parameters can be found in `cli_params_emnist.txt` and `cli_params_shakespeare.txt` for EMNIST and Shakespeare tasks, respectively.

You can monitor the progress using TensorBoard:

```setup
tensorboard --logdir <root_output_dir>/logdir
```

## Results

Execute `plots.ipynb` using [Jupyter](https://jupyter.org/) to re-create figures 3 and 9 from the paper. 