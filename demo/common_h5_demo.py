import sys

sys.path.append('../')
import os
import utils.data_utils as datasets
import argparse
from utils.features import FeatureMap
from utils.utils import set_logger, print_to_json, load_config
import logging
from utils.torch_utils import seed_everything
import model

if __name__ == '__main__':
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dataset_yaml', type=str, default='../config/dataset_config/avazu.yaml')
    parser.add_argument('--config_model_yaml', type=str, default='../config/model_config/CINFM2.yaml')
    parser.add_argument('--experiment_id', type=str, default='CINFM2_base')
    parser.add_argument('--dataset_id', type=str, default='avazu_1w_demo')
    args = vars(parser.parse_args())

    # 读取配置文件信息
    params = load_config(args['config_model_yaml'], args['experiment_id'], args['config_dataset_yaml'],
                         args['dataset_id'])

    set_logger(params)  # 打印日志信息
    logging.info('Start the demo...')
    logging.info(print_to_json(params))  # 打印参数
    seed_everything(seed=params['seed'])  # 设置随机数种子，固定每一次的训练结果。随机数种子seed确定时，模型的训练结果将始终保持一致。

    # Load feature_map from json
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(os.path.join(data_dir, "feature_map.json"))

    train_gen, valid_gen = datasets.h5_generator(feature_map,
                                                 stage='train',
                                                 train_data=os.path.join(data_dir, 'train.h5'),
                                                 valid_data=os.path.join(data_dir, 'valid.h5'),
                                                 batch_size=params['batch_size'],
                                                 shuffle=params['shuffle'])

    # Model initialization and fitting  动态加载模型
    model_class = getattr(model, params['model_name'])
    model = model_class(feature_map, **params)
    model.count_parameters()  # print number of parameters used in model
    # model.fit_generator(train_gen,
    #                     validation_data=valid_gen,
    #                     epochs=params['epochs'],
    #                     verbose=params['verbose'])

    model.load_weights(model.checkpoint)  # reload the best checkpoint

    logging.info('***** validation results *****')
    # model.evaluate_generator(valid_gen)

    logging.info('***** test results *****')
    test_gen = datasets.h5_generator(feature_map,
                                     stage='test',
                                     test_data=os.path.join(data_dir, 'test.h5'),
                                     batch_size=params['batch_size'],
                                     shuffle=False)
    model.evaluate_generator(test_gen)
