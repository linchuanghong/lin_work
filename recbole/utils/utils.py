# -*- coding: utf-8 -*-
# @Time   : 2020/7/17
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2021/3/8
# @Author : Jiawei Guan
# @Email  : guanjw@ruc.edu.cn

"""
recbole.utils.utils
################################
"""

import datetime
import importlib
import os
import random
import math

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from recbole.utils.enum_type import ModelType
from recbole.utils.groupnorm import ComplexGroupNorm


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur


def ensure_dir(dir_path):
    r"""Make sure the directory exists, if it does not exist, create it

    Args:
        dir_path (str): directory path

    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_model(model_name):
    r"""Automatically select model class based on model name

    Args:
        model_name (str): model name

    Returns:
        Recommender: model class
    """
    model_submodule = [
        'general_recommender', 'context_aware_recommender', 'sequential_recommender', 'knowledge_aware_recommender',
        'exlib_recommender'
    ]

    model_file_name = model_name.lower()
    model_module = None
    for submodule in model_submodule:
        module_path = '.'.join(['recbole.model', submodule, model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            break

    if model_module is None:
        raise ValueError('`model_name` [{}] is not the name of an existing model.'.format(model_name))
    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer(model_type, model_name):
    r"""Automatically select trainer class based on model type and model name

    Args:
        model_type (ModelType): model type
        model_name (str): model name

    Returns:
        Trainer: trainer class
    """
    try:
        return getattr(importlib.import_module('recbole.trainer'), model_name + 'Trainer')
    except AttributeError:
        if model_type == ModelType.KNOWLEDGE:
            return getattr(importlib.import_module('recbole.trainer'), 'KGTrainer')
        elif model_type == ModelType.TRADITIONAL:
            return getattr(importlib.import_module('recbole.trainer'), 'TraditionalTrainer')
        else:
            return getattr(importlib.import_module('recbole.trainer'), 'Trainer')


def early_stopping(value, best, cur_step, max_step, bigger=True):
    r""" validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value > best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def calculate_valid_score(valid_result, valid_metric=None):
    r""" return valid score from valid result

    Args:
        valid_result (dict): valid result
        valid_metric (str, optional): the selected metric in valid result for valid score

    Returns:
        float: valid score
    """
    if valid_metric:
        return valid_result[valid_metric]
    else:
        return valid_result['Recall@10']


def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric) + ' : ' + str(value) + '    '
    return result_str


def init_seed(seed, reproducibility):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_tensorboard(logger):
    r""" Creates a SummaryWriter of Tensorboard that can log PyTorch models and metrics into a directory for 
    visualization within the TensorBoard UI.
    For the convenience of the user, the naming rule of the SummaryWriter's log_dir is the same as the logger.

    Args:
        logger: its output filename is used to name the SummaryWriter's log_dir.
                If the filename is not available, we will name the log_dir according to the current time.

    Returns:
        SummaryWriter: it will write out events and summaries to the event file.
    """
    base_path = 'log_tensorboard'

    dir_name = None
    for handler in logger.handlers:
        if hasattr(handler, "baseFilename"):
            dir_name = os.path.basename(getattr(handler, 'baseFilename')).split('.')[0]
            break
    if dir_name is None:
        dir_name = '{}-{}'.format('model', get_local_time())

    dir_path = os.path.join(base_path, dir_name)
    writer = SummaryWriter(dir_path)
    return writer


def get_gpu_usage(device=None):
    r""" Return the reserved memory and total memory of given device in a string.
    Args:
        device: cuda.device. It is the device that the model run on.

    Returns:
        str: it contains the info about reserved memory and total memory of given device.
    """

    reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
    total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3

    return '{:.2f} G/{:.2f} G'.format(reserved, total)

class SimpleRetention(nn.Module):
    def __init__(self, hidden_size, gamma, precision="single"):
        """
        Simple retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(SimpleRetention, self).__init__()

        if precision == "half":
            raise NotImplementedError("batchmm does not support half precision complex yet.")
            self.complex_type = torch.complex32
            self.real_type = torch.float16
        elif precision == "single":
            self.complex_type = torch.complex64
            self.real_type = torch.float32

        self.precision = precision
        self.hidden_size = hidden_size
        self.gamma = gamma

        self.i = torch.complex(torch.tensor(0.0), torch.tensor(1.0))

        self.W_Q = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=self.real_type) / hidden_size)
        self.W_K = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=self.real_type) / hidden_size)
        self.W_V = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=self.real_type) / hidden_size)
        

        self.theta = torch.randn(hidden_size) / hidden_size
        self.theta = nn.Parameter(self.theta)

        

    def forward(self, X):
        """
        Parallel (default) representation of the retention mechanism.
        X: (batch_size, sequence_length, hidden_size)
        """
        sequence_length = X.shape[1]
        D = self._get_D(sequence_length)

        if X.dtype != self.complex_type:
            X = torch.complex(X, torch.zeros_like(X)).to(self.complex_type)
        
        i = self.i.to(X.device)
        ns = torch.arange(1, sequence_length + 1, dtype=self.real_type, device=X.device)
        ns = torch.complex(ns, torch.zeros_like(ns)).to(self.complex_type)
        Theta = []

        for n in ns:
            Theta.append(torch.exp(i * n * self.theta))
        
        Theta = torch.stack(Theta, dim=0)
        
        Theta_bar = Theta.conj()

        Q = (X @ self.W_Q.to(self.complex_type)) * Theta.unsqueeze(0)
        K = (X @ self.W_K.to(self.complex_type)) * Theta_bar.unsqueeze(0)
        V = X @ self.W_V.to(self.complex_type)
        att = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0)
        
        return att @ V
        
    def forward_recurrent(self, x_n, s_n_1, n):
        """
        Recurrent representation of the retention mechanism.
        x_n: (batch_size, hidden_size)
        s_n_1: (batch_size, hidden_size)
        """
        if x_n.dtype != self.complex_type:
            x_n = torch.complex(x_n, torch.zeros_like(x_n)).to(self.complex_type)
        
        n = torch.tensor(n, dtype=self.complex_type, device=x_n.device)

        Theta = torch.exp(self.i * n * self.theta)
        Theta_bar = Theta.conj()

        Q = (x_n @ self.W_Q.to(self.complex_type)) * Theta
        K = (x_n @ self.W_K.to(self.complex_type)) * Theta_bar
        V = x_n @ self.W_V.to(self.complex_type)

        # K: (batch_size, hidden_size)
        # V: (batch_size, hidden_size)
        # s_n_1: (batch_size, hidden_size, hidden_size)
        # s_n = gamma * s_n_1 + K^T @ V

        s_n = self.gamma * s_n_1 + K.unsqueeze(2) @ V.unsqueeze(1)
        
        return (Q.unsqueeze(1) @ s_n).squeeze(1), s_n
    
    def _get_D(self, sequence_length):
        # D[n,m] = gamma ** (n - m) if n >= m else 0
        D = torch.zeros((sequence_length, sequence_length), dtype=self.real_type, requires_grad=False)
        for n in range(sequence_length):
            for m in range(sequence_length):
                if n >= m:
                    D[n, m] = self.gamma ** (n - m)
        return D.to(self.complex_type)

class MultiScaleRetention(nn.Module):
    def __init__(self, hidden_size, heads, precision="single"):
        """
        Multi-scale retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(MultiScaleRetention, self).__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.precision = precision
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads

        if precision == "half":
            raise NotImplementedError("batchmm does not support half precision complex yet.")
            self.complex_type = torch.complex32
            self.real_type = torch.float16
        elif precision == "single":
            self.complex_type = torch.complex64
            self.real_type = torch.float32
        
        self.gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), heads, dtype=self.real_type))).detach().cpu().tolist()

        self.swish = lambda x: x * torch.sigmoid(x)
        self.W_G = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=self.complex_type) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=self.complex_type) / hidden_size)
        self.group_norm = ComplexGroupNorm(heads, hidden_size)

        self.retentions = nn.ModuleList([
            SimpleRetention(self.head_size, gamma) for gamma in self.gammas
        ])

    def forward(self, X):
        """
        parallel representation of the multi-scale retention mechanism
        """
        if X.dtype != self.complex_type:
            X = torch.complex(X, torch.zeros_like(X)).to(self.complex_type)
        
        # apply each individual retention mechanism to a slice of X
        Y = []
        for i in range(self.heads):
            Y.append(self.retentions[i](X[:, :, i*self.head_size:(i+1)*self.head_size]))
        
        Y = torch.cat(Y, dim=2)
        Y = self.group_norm(Y.reshape(-1, self.hidden_size)).reshape(X.shape)

        return (self.swish(X @ self.W_G.to(self.complex_type)) * Y) @ self.W_O.to(self.complex_type)
    
    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        recurrent representation of the multi-scale retention mechanism
        """
        if x_n.dtype != self.complex_type:
            x_n = torch.complex(x_n, torch.zeros_like(x_n)).to(self.complex_type)
        n = torch.tensor(n, dtype=self.complex_type, device=x_n.device)

        # apply each individual retention mechanism to a slice of X
        Y = []
        s_ns = []
        for i in range(self.heads):
            y, s_n = self.retentions[i].forward_recurrent(
                x_n[:, i*self.head_size:(i+1)*self.head_size], s_n_1s[i], n
                )
            Y.append(y)
            s_ns.append(s_n)
        
        Y = torch.cat(Y, dim=1)
        Y = self.group_norm(Y)
        return (self.swish(x_n @ self.W_G.to(self.complex_type)) * Y) @ self.W_O.to(self.complex_type), s_ns

