#!/bin/bash

cd /home/user/Documents/Baptiste/surrogate_modelling/gns

DATASET_NAME="granular_collapse"
DATA_PATH="examples/${DATASET_NAME}/datasets/"
MODEL_PATH="examples/${DATASET_NAME}/models/"

MODEL_FILE="model-12000.pt"


python -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --model_file=${MODEL_FILE} --mode="valid"