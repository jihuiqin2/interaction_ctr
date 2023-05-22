import sys

sys.path.append('../')
import os
import utils.data_utils as datasets
from datasets.criteo import FeatureEncoder
from utils.utils import set_logger, print_to_json, load_dataset_config
import logging

if __name__ == '__main__':
    config_dataset_yaml = '../config/dataset_config/criteo.yaml'
    dataset_id = 'criteo_1w_demo'
    params = load_dataset_config(config_dataset_yaml, dataset_id)

    # set up logger and random seed
    set_logger(params, log_file='../checkpoints/criteo.log')
    logging.info(print_to_json(params))

    feature_encoder = FeatureEncoder(feature_cols=params["feature_cols"],
                                     label_col=params["label_col"],
                                     dataset_id=dataset_id,
                                     data_root=params["data_root"])

    datasets.build_dataset(feature_encoder,
                           all_data=params["all_data"] if ("all_data" in params) else None,
                           train_data=params["train_data"] if ("train_data" in params) else None,
                           valid_data=params["valid_data"] if ("valid_data" in params) else None,
                           test_data=params["test_data"] if ("test_data" in params) else None,
                           valid_size=params["valid_size"],
                           test_size=params["test_size"])
