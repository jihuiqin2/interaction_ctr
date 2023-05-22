# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import sys

sys.path.append('../')
import os
import utils.data_utils as datasets
from utils.features import FeatureMap
from utils.utils import set_logger, print_to_json, load_config
import logging
from model.AFM import AFM
from utils.torch_utils import seed_everything

if __name__ == '__main__':
    # 读取配置文件信息
    config_model_yaml = '../config/model_config/AFM.yaml'
    experiment_id = 'AFM_base'  # 加载哪个模型

    # config_dataset_yaml = '../config/dataset_config/tiny_data.yaml'   # 使用哪个数据集
    # config_dataset_yaml = '../config/dataset_config/criteo.yaml'
    config_dataset_yaml = '../config/dataset_config/avazu.yaml'

    params = load_config(config_model_yaml, experiment_id, config_dataset_yaml)

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
                                                 train_data=os.path.join(data_dir, 'train_part_*'),
                                                 valid_data=os.path.join(data_dir, 'valid_part_*'),
                                                 batch_size=params['batch_size'],
                                                 shuffle=params['shuffle'],
                                                 data_block_size=params['data_block_size']
                                                 )
    model = AFM(feature_map, **params)
    model.fit_generator(train_gen, validation_data=valid_gen, epochs=params['epochs'],
                        verbose=params['verbose'])  # 开始训练
    # 加载模型
    model.load_weights(model.checkpoint)

    logging.info('***** validation results *****')
    model.evaluate_generator(valid_gen)

    logging.info('***** test results *****')
    test_gen = datasets.h5_generator(feature_map,
                                     stage='test',
                                     test_data=os.path.join(data_dir, 'test_part_*'),
                                     batch_size=params['batch_size'],
                                     shuffle=False)
    model.evaluate_generator(test_gen)
