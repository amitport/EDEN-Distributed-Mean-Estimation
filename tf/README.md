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

You can monitor the progress using TensorBoard:

```setup
tensorboard --logdir <root_output_dir>/logdir
```

## Results

Execute `plots.ipynb` using [Jupyter](https://jupyter.org/) to re-create figures 3 and 9 from the paper. 