# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (C) 2021. The Chinese University of Hong Kong. All rights reserved.
#
# Authors: Jinyang Liu <The Chinese University of Hong Kong>
#          Jieming Zhu <Huawei Noah's Ark Lab>
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


import torch.nn as nn
import numpy as np
import torch
import os, sys
import logging
from utils.metrics import evaluate_metrics
from utils.torch_utils import get_device, get_optimizer, get_loss_fn, get_regularizer
from utils.utils import Monitor
import time


class BaseModel(nn.Module):
    def __init__(self,
                 feature_map,
                 model_id="BaseModel",
                 gpu=-1,
                 monitor="AUC",
                 save_best_only=True,
                 monitor_mode="max",
                 patience=2,
                 every_x_epochs=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 reduce_lr_on_plateau=True,  # todo  作用
                 rand_number='',
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)",
                 **kwargs):
        super(BaseModel, self).__init__()
        self.device = get_device(gpu)  # 使用cpu还是gpu
        self._monitor = Monitor(kv=monitor)
        self._monitor_mode = monitor_mode
        self._patience = patience
        self._every_x_epochs = every_x_epochs  # float acceptable
        self._save_best_only = save_best_only
        self._embedding_regularizer = embedding_regularizer
        self._net_regularizer = net_regularizer
        self._reduce_lr_on_plateau = reduce_lr_on_plateau
        self._embedding_initializer = embedding_initializer
        self._feature_map = feature_map
        self.model_id = model_id
        self.model_dir = os.path.join(kwargs["model_root"], feature_map.dataset_id)
        self.checkpoint = os.path.abspath(
            os.path.join(self.model_dir, self.model_id + str(rand_number) + ".model"))
        self._validation_metrics = kwargs["metrics"]
        self._verbose = kwargs["verbose"]

    # 定义优化器和损失函数
    def compile(self, optimizer, loss, lr):
        self.optimizer = get_optimizer(optimizer, self.parameters(), lr)
        self.loss_fn = get_loss_fn(loss)

    # 损失函数通常由损失项和正则化项组成
    def get_total_loss(self, inputs):
        total_loss = self.add_loss(inputs) + self.add_regularization()
        return total_loss

    # 损失函数
    def add_loss(self, inputs, reduction="mean"):
        return_dict = self.forward(inputs)
        loss = self.loss_fn(return_dict["y_pred"], return_dict["y_true"], reduction=reduction)
        return loss

    # 损失函数的正则化
    def add_regularization(self):
        reg_loss = 0
        if self._embedding_regularizer or self._net_regularizer:
            emb_reg = get_regularizer(self._embedding_regularizer)
            net_reg = get_regularizer(self._net_regularizer)
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if "embedding_layer" in name:
                        if self._embedding_regularizer:
                            for emb_p, emb_lambda in emb_reg:
                                reg_loss += (emb_lambda / emb_p) * torch.norm(param, emb_p) ** emb_p
                    else:
                        if self._net_regularizer:
                            for net_p, net_lambda in net_reg:
                                reg_loss += (net_lambda / net_p) * torch.norm(param, net_p) ** net_p
        return reg_loss

    # 参数初始化
    def reset_parameters(self):
        def reset_param(m):
            if type(m) == nn.ModuleDict:
                for k, v in m.items():
                    if type(v) == nn.Embedding:
                        if "pretrained_emb" in self._feature_map.feature_specs[k]:  # skip pretrained
                            continue
                        if self._embedding_initializer is not None:
                            try:
                                if v.padding_idx is not None:
                                    # the last index is padding_idx
                                    initializer = self._embedding_initializer.replace("(", "(v.weight[0:-1, :],")
                                else:
                                    initializer = self._embedding_initializer.replace("(", "(v.weight,")
                                eval(initializer)
                            except:
                                raise NotImplementedError("embedding_initializer={} is not supported." \
                                                          .format(self._embedding_initializer))
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)

        self.apply(reset_param)

    # 数据使用cpu还是gpu
    def inputs_to_device(self, inputs):
        X, y = inputs
        X = X.to(self.device)
        y = y.float().view(-1, 1).to(self.device)
        self.batch_size = y.size(0)
        return X, y

    # 模型 todo 使用多个GPU
    def model_to_device(self):
        if torch.cuda.device_count() >= 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # self = nn.DataParallel(self)
        self.to(device=self.device)

    # 调整学习率
    def lr_decay(self, factor=0.1, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            reduced_lr = max(param_group["lr"] * factor, min_lr)
            param_group["lr"] = reduced_lr
        return reduced_lr

    # 模型训练
    def fit_generator(self, data_generator, epochs=1, validation_data=None,
                      verbose=0, max_gradient_norm=10., **kwargs):
        """
        Training a model and valid accuracy.
        Inputs:
        - iter_train: I
        - iter_val: .
        - optimizer: Abstraction of optimizer used in training process, e.g., "torch.optim.Adam()""torch.optim.SGD()".
        - epochs: Integer, number of epochs.
        - max_gradient_norm: to address 梯度爆炸问题
        - verbose: Bool, if print.
        """
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._total_batches = 0
        self._batches_per_epoch = len(data_generator)  # 总样本数/batch_size，1个epoch需要几轮训练
        self._every_x_batches = int(np.ceil(self._every_x_epochs * self._batches_per_epoch))
        self._stop_training = False
        self._verbose = verbose

        logging.info("Start training: {} batches/epoch".format(self._batches_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            epoch_loss = self.train_one_epoch(data_generator, epoch)
            logging.info("Train loss: {:.6f}".format(epoch_loss))
            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(epoch + 1))
        logging.info("Training finished.")

    def train_one_epoch(self, data_generator, epoch):
        epoch_loss = 0
        self.train()
        batch_iterator = data_generator
        if self._verbose > 0:  # 显示进度条
            from tqdm import tqdm
            # 可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)。
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_iterator):
            self.optimizer.zero_grad()
            loss = self.get_total_loss(batch_data)
            loss.backward()
            # 梯度剪裁，当梯度小于阈值时，更新的梯度为阈值  https://blog.csdn.net/zhaohongfei_358/article/details/122820992
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            epoch_loss += loss.item()  # 获取tensor的值
            self.on_batch_end(batch_index)
            if self._stop_training:
                break
        return epoch_loss / self._batches_per_epoch

    def on_batch_end(self, batch, logs={}):  #
        self._total_batches += 1
        if (batch + 1) % self._every_x_batches == 0 or (batch + 1) % self._batches_per_epoch == 0:  # 每个epoch的所有轮次训练结束后
            epoch = round(float(self._total_batches) / self._batches_per_epoch, 2)  # 第几个epoch，小数点保留1位
            val_logs = self.evaluate_generator(self.valid_gen)  # 使用验证集进行评估
            self.checkpoint_and_earlystop(epoch, val_logs)
            logging.info("--- {}/{} batches finished ---".format(batch + 1, self._batches_per_epoch))

    # 保存模型
    def checkpoint_and_earlystop(self, epoch, logs, min_delta=1e-6):
        monitor_value = self._monitor.get_value(logs)  #
        logging.info(
            "[compute] monitor_value: {}, compare data: {}".format(monitor_value, self._best_metric - min_delta))
        if (self._monitor_mode == "min" and monitor_value > self._best_metric - min_delta) or \
                (self._monitor_mode == "max" and monitor_value < self._best_metric + min_delta):
            self._stopping_steps += 1
            logging.info("Monitor({}) STOP: {:.6f} !".format(self._monitor_mode, monitor_value))
            if self._reduce_lr_on_plateau:
                current_lr = self.lr_decay()
                logging.info("Reduce learning rate on plateau: {:.6f}".format(current_lr))  # 学习率
        else:
            self._stopping_steps = 0
            self._best_metric = monitor_value
            if self._save_best_only:
                logging.info("Save best model: monitor({}): {:.6f}" \
                             .format(self._monitor_mode, monitor_value))
                self.save_weights(self.checkpoint)
        if self._stopping_steps * self._every_x_epochs >= self._patience:  # 结束训练的条件
            self._stop_training = True
            logging.info("Early stopping at epoch={:g}".format(epoch))
        if not self._save_best_only:
            self.save_weights(self.checkpoint)

    # 只保存模型参数
    def save_weights(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

    # 模型评估（验证集）
    def evaluate_generator(self, data_generator):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            y_true = []
            if self._verbose > 0:
                from tqdm import tqdm
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))  # extend累计
                y_true.extend(batch_data[1].data.cpu().numpy().reshape(-1))

            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            val_logs = self.evaluate_metrics(y_true, y_pred, self._validation_metrics)

            return val_logs  # 验证集的，logloss: 0.640982 - AUC: 0.598378

    def evaluate_metrics(self, y_true, y_pred, metrics):
        return evaluate_metrics(y_true, y_pred, metrics)

    # 获取预测值
    def predict_generator(self, data_generator):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            if self._verbose > 0:
                from tqdm import tqdm
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            return y_pred

    # 加载模型
    def load_weights(self, checkpoint):
        self.to(self.device)
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state_dict)
        del state_dict
        torch.cuda.empty_cache()

    # 使用激活函数
    def get_output_activation(self, task="binary_classification"):
        if task == "binary_classification":
            return nn.Sigmoid()
        elif task == "regression":
            return None
        else:
            raise NotImplementedError("task={} is not supported.".format(task))

    # 训练的参数个数
    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters():
            if not count_embedding and "embedding" in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        logging.info("Total number of parameters: {}.".format(total_params))
