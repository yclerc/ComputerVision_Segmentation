# Non destructive Testing with Ultrasonic Images


## Description

This deep learning algorithm generates segmentation based on iput images and masks. 

## Getting Started

### Dependencies

See requirements.txt

### Installing

Clone and go to the newly created repository :

    $ git clone <https address>
    $ cd <Path to project directory>

Create a virtualenv and activate it: 

    $ python -m venv <my virtual env>
    $ source <my virtual env>/bin/activate

Or on Windows cmd:

    $ python -m venv <my virtual env>
    $ <my virtual env>\Scripts\activate

Install requirements from txt file:

    $ pip install -r requirements.txt


### Project structure

Folders:

- data/: dataset used, containing images and masks subfolders.
- files/: contains a CSV file and model weights generated while training the model.
- logs/: contains TensorBoard log files.
- results/: used to store the results inferred.

Scripts:
- data.py: code for loading the dataset, reading the images and masks. Creates a tf.data pipeline for training, validation and testing datasets.
- model.py: code for the UNet architecture. 
- train.py: trains the model from scratch with dataset located in data/.   
- predict.py: generates inferences on test images from model saved after running train.py


### Executing program

To start generating inferences:

    $ python train.py


To launch a new training of the model: 

    $ python predict.py



To review training logs in TensorBoard

    $ tensorboard --logdir=./logs/




## Authors

Contributors names and contact info

- Maria ZAVLYANOVA
- Victor LAMBART
- Yann CLERC
