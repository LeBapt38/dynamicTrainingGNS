#!/bin/bash

cd /home/user/Documents/Baptiste/surrogate_modelling/gns


# Train for a few steps on these data.
DATASET_NAME="granular_collapse"
DATA_PATH="examples/${DATASET_NAME}/datasets/"
MODEL_PATH="examples/${DATASET_NAME}/models/"

# The training restart at this point :
MODEL_FILE="model-2200.pt"
TRAIN_STATE_FILE="train_state-2200.pt"
NTRAINING_STEPS=2300

python3 -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --model_file=${MODEL_FILE} --train_state_file=${TRAIN_STATE_FILE} --ntraining_steps=${NTRAINING_STEPS}