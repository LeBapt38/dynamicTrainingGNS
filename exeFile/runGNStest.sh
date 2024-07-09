#!/bin/bash

cd /home/user/Documents/Baptiste/surrogate_modelling/gns

DATASET_NAME="granular_collapse_2d"
DATA_PATH="examples/${DATASET_NAME}/datasets/"
MODEL_PATH="examples/${DATASET_NAME}/models/"
ROLLOUT_PATH="examples/${DATASET_NAME}/rollouts/"

MODEL_FILE="model-400000.pt"


python -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --model_file=${MODEL_FILE} --output_path=${ROLLOUT_PATH} --mode='rollout'
