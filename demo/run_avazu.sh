#!/usr/bin/env bash
configDataset='../config/dataset_config/avazu.yaml'
configModel='../config/model_config/CINFM.yaml'
experimentId='CINFM_base'

python common_h5_demo.py \
  --config_model_yaml $configModel \
  --config_dataset_yaml $configDataset \
  --experiment_id $experimentId

python common_h5_demo.py --config_model_yaml='../config/model_config/CINFM.yaml' --config_dataset_yaml='../config/dataset_config/avazu.yaml' --experiment_id='CINFM_base'


python common_h5_demo.py --config_model_yaml='../config/model_config/CINFM.yaml' --config_dataset_yaml='../config/dataset_config/avazu.yaml' --experiment_id='CINFM_base'
