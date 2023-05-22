#!/usr/bin/env bash
configDataset='../config/dataset_config/criteo.yaml'
configModel='../config/model_config/LR.yaml'
experimentId='LR_base'

python common_h5_demo.py \
  --config_model_yaml $configModel \
  --config_dataset_yaml $configDataset \
  --experiment_id $experimentId
