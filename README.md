# CAD Predictor
Scripts for running Classroom Activity Detection (teacher, student, multiple) over an input video.

## Installing

To install this package and all the required dependencies, make sure you've some version of python 3.10 installed, then clone the repo and run:

```shell
$ cd cad-predictor
$ poetry env use 3.10
$ poetry install
```

## Usage

Before running any command, make sure to activate the virtualenv where the dependencies are installed:

```shell
$ cd cad-predictor
$ poetry shell
```

To run the LSTM model (default), download the model weights and save them as `lstm.ckpt`, and run:

```shell
$ python predict.py path/to/input/video.mp4
```

