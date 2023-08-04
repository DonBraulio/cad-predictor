# CAD Predictor

Scripts for running Classroom Activity Detection (teacher, student, multiple) over an input video.

## Installing

To install this package and all the required dependencies, make sure you've some version of python 3.10 installed, then clone the repo and run:

```shell
$ cd cad-predictor

# Optional: create and activate virtual environment
$ python -m venv .venv/
$ . .venv/bin/activate

# Install dependencies
$ pip install -r requirements.txt
```

## Usage

```shell
$ cd cad-predictor

# Activate virtualenv if you created it above (optional)
$ . .venv/bin/activate

# Run on provided sample or any other path/to/video.mp4
$ python predict.py sample.mp4

# Create an output video 😎
$ python predict.py sample.mp4 --create_video
```
