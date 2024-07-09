#!/bin/bash

cd /home/user/Documents/Baptiste/surrogate_modelling/gns

# Download a sample of a dataset.
DATASET_NAME="granular_collapse_2d"

# Train for a few steps.
DATA_PATH="examples/${DATASET_NAME}/datasets/"
MODEL_PATH="examples/${DATASET_NAME}/models/"
NTRAINING_STEPS=100

python -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --output_path=${MODEL_PATH} --ntraining_steps=${NTRAINING_STEPS} --mode='train'