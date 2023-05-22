# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import os
import logging
import logging.config
import yaml
import glob
import time
import json
from collections import OrderedDict


# 加载模型参数
def load_config(config_model_yaml, experiment_id, config_dataset_yaml, dataset_id):
    params = dict()
    model_configs = glob.glob(os.path.join(config_model_yaml))
    if not model_configs:
        raise RuntimeError('config_dir={} is not valid!'.format(config_model_yaml))
    found_params = dict()
    for config in model_configs:
        with open(config, 'rb') as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)  # 读取文件，得到字典数据
            if 'Base' in config_dict:
                found_params['Base'] = config_dict['Base']
            if experiment_id in config_dict:
                found_params[experiment_id] = config_dict[experiment_id]
        if len(found_params) == 2:
            break
    if experiment_id not in found_params:
        raise ValueError("expid={} not found in config".format(experiment_id))
    params.update(found_params.get('Base', {}))
    params.update(found_params.get(experiment_id))
    params['model_id'] = experiment_id
    params['dataset_id'] = dataset_id
    dataset_params = load_dataset_config(config_dataset_yaml, params['dataset_id'])
    params.update(dataset_params)
    return params


# 加载数据集参数
def load_dataset_config(config_dataset_yaml, dataset_id):
    dataset_configs = glob.glob(os.path.join(config_dataset_yaml))
    if not dataset_configs:
        raise RuntimeError('config_dir={} is not valid!'.format(dataset_configs))
    for config in dataset_configs:
        with open(config, 'rb') as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if dataset_id in config_dict:
                return config_dict[dataset_id]
    raise RuntimeError('dataset_id={} is not found in config.'.format(dataset_id))


# 打印日志
def set_logger(params, log_file=None):
    if log_file is None:
        dataset_id = params['dataset_id']
        model_id = params['model_id']
        log_dir = os.path.join(params['model_root'], dataset_id)
        log_file = os.path.join(log_dir, model_id + str(params['rand_number']) + '.log')
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    '''
    %(asctime)s: 打印日志的时间，%(process)d: 打印进程ID，%(levelname)s: 打印日志级别名称， %(message)s: 打印日志信息
    %(levelno)s: 打印日志级别的数值， %(lineno)d: 打印日志的当前行号，
    '''
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s P%(process)d %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler()])


def print_to_json(data, sort_keys=True):
    new_data = dict((k, str(v)) for k, v in data.items())
    if sort_keys:
        new_data = OrderedDict(sorted(new_data.items(), key=lambda x: x[0]))
    return json.dumps(new_data, indent=4)


def print_to_list(data):
    return ' - '.join('{}: {:.6f}'.format(k, v) for k, v in data.items())


class Monitor(object):
    def __init__(self, kv):
        if isinstance(kv, str):
            kv = {kv: 1}
        self.kv_pairs = kv  # { 'AUC': 1, 'logloss': -1 }

    def get_value(self, logs):  # 验证集中，对象logloss和auc
        value = 0
        for k, v in self.kv_pairs.items():
            value += logs.get(k, 0) * v
        return value  # (logs.logloss*-1)+(logs.auc*1)
