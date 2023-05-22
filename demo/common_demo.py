import sys

sys.path.append('../')
import os
import utils.data_utils as datasets
# from datasets.taobao import FeatureEncoder
# from datasets.criteo import FeatureEncoder
from datasets.avazu import FeatureEncoder
from utils.utils import set_logger, print_to_json, load_config
import logging
from model.AutoInt import AutoInt
from utils.torch_utils import seed_everything

if __name__ == '__main__':
    # 读取配置文件信息
    config_model_yaml = '../config/model_config/AutoInt.yaml'
    experiment_id = 'AutoInt_base'  # 加载哪个模型
    # config_dataset_yaml = '../config/dataset_config/tiny_data.yaml'
    # config_dataset_yaml = '../config/dataset_config/criteo.yaml'
    config_dataset_yaml = '../config/dataset_config/avazu.yaml'

    params = load_config(config_model_yaml, experiment_id, config_dataset_yaml)

    set_logger(params)  # 打印日志信息
    logging.info('Start the demo...')
    logging.info(print_to_json(params))  # 打印参数
    seed_everything(seed=params['seed'])  # 设置随机数种子，固定每一次的训练结果。随机数种子seed确定时，模型的训练结果将始终保持一致。

    # Set feature_encoder that defines how to preprocess data
    feature_encoder = FeatureEncoder(**params)

    # Build dataset from csv to h5
    datasets.build_dataset(feature_encoder,
                           all_data=params["all_data"] if ("all_data" in params) else None,
                           train_data=params["train_data"] if ("train_data" in params) else None,
                           valid_data=params["valid_data"] if ("valid_data" in params) else None,
                           test_data=params["test_data"] if ("test_data" in params) else None,
                           valid_size=params["valid_size"],
                           test_size=params["test_size"])

    # Get feature_map that defines feature specs
    feature_map = feature_encoder.feature_map

    # Get train and validation data generator from h5
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    train_gen, valid_gen = datasets.h5_generator(feature_map,
                                                 stage='train',
                                                 train_data=os.path.join(data_dir, 'train.h5'),
                                                 valid_data=os.path.join(data_dir, 'valid.h5'),
                                                 batch_size=params['batch_size'],
                                                 shuffle=params['shuffle'])

    # Model initialization and fitting
    model = AutoInt(feature_encoder.feature_map, **params)
    model.count_parameters()  # print number of parameters used in model
    model.fit_generator(train_gen,
                        validation_data=valid_gen,
                        epochs=params['epochs'],
                        verbose=params['verbose'])
    model.load_weights(model.checkpoint)  # reload the best checkpoint

    logging.info('***** validation results *****')
    model.evaluate_generator(valid_gen)

    logging.info('***** test results *****')
    test_gen = datasets.h5_generator(feature_map,
                                     stage='test',
                                     test_data=os.path.join(data_dir, 'test.h5'),
                                     batch_size=params['batch_size'],
                                     shuffle=False)
    model.evaluate_generator(test_gen)

#
