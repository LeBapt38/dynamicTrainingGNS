#!/bin/bash

cd /home/user/Documents/Baptiste/surrogate_modelling/gns

DATASET_NAME="granular_collapse"
DATA_PATH="examples/${DATASET_NAME}/datasets/"
MODEL_PATH="examples/${DATASET_NAME}/models/"
ROLLOUT_PATH="examples/${DATASET_NAME}/rollouts/"
mkdir -p ${ROLLOUT_PATH}

MODEL_FILE="model-0.pt"
OUTPUT_MODE="vtk"

python -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --model_file=${MODEL_FILE} --output_path=${ROLLOUT_PATH} --mode='rollout'

# Plot the first rollout.
python -m gns.render_rollout --output_mode="${OUTPUT_MODE}"  --rollout_dir="${ROLLOUT_PATH}" --rollout_name="rollout_ex0" 


