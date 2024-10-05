# Deepweeds computer vision modelling

This was a fun little side project to learn a bit of Keras (coming from a pytorch background) and train a boat load of models.

The task is the classic image single-label classification scenario where we have 9 sets of weed variants and need to predict which, if any, are present in a picture.  A couple of challenges with this dataset were 1. image quality is quite poor and has bad leakage with background objects / camera rotation etc, and 2. there is a big imbalance towards the negative class (class with no weeds).  Have a look at the `notebooks` folder for the light EDA workings.

The data can be accessed via tensorflow datasets - https://www.tensorflow.org/datasets/catalog/deep_weeds

## Running locally

Please note that we use git lfs for training artifacts (anything under `outputs/` or `multirun/`), so before running make sure you have `git-lfs` installed, and the artifacts cloned locally.

```sh
git lfs fetch --all
git lfs pull
```

### MacOS 14+, Linux w/CUDA, Windows Subsystem for Linux V2

1. Install `uv` as this is how we will manage the project dependencies

```sh
brew install uv
```

or

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a local environment

```sh
uv sync
```

3. Manually run `notebooks/01_data_exploration.ipynb` to generate our local dataset

4. Now you're ready to run experiments, try changing some model or training parameters in `conf/config.yaml` then start a new training job

```sh
uv run python src/deepweeds/train.py
```

5. You can check the training artifacts such as exact input configuration, stdout log, etc. by finding the date/time folder under `outputs/` corresponding to when you started a run

6. Evaluate results through tensorboard if you'd like to compare against past runs

```sh
uv run tensorboard --logdir=outputs
```

or if you want to look at one of the grid search runs

```sh
uv run tensorboard --logdir=multirun
```

### MacOS <14, Linux with TPU / non-cuda gpus, navive Windows, WSL V1

The code is not guaranteed in any way to run as intended under this configuration due to various issues like MacOS 13 not supporting a few required tensor ops on the metal gpu, tensorflow not having a cuda enabled distribution for windows etc.

So try at your own risk...
