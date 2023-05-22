import sys

sys.path.append('../')
import utils.data_utils as datasets
from datasets.avazu import FeatureEncoder
from utils.utils import set_logger, print_to_json, load_dataset_config
import logging

if __name__ == '__main__':
    dataset_id = 'avazu_1w_demo'
    config_dataset_yaml = '../config/dataset_config/avazu.yaml'
    params = load_dataset_config(config_dataset_yaml, dataset_id)

    # set up logger and random seed
    set_logger(params, log_file='../checkpoints/avazu.log')
    logging.info(print_to_json(params))

    feature_encoder = FeatureEncoder(feature_cols=params["feature_cols"],
                                     label_col=params["label_col"],
                                     dataset_id=dataset_id,
                                     data_root=params["data_root"])

    datasets.build_dataset(feature_encoder,
                           **params)
