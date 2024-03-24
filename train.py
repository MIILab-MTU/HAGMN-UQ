from encodings import normalize_encoding
from locale import normalize
import os
import sys
import time
import random
import pandas as pd
import shutil
import networkx as nx
import argparse
import json

from typing import List

import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

from methods.HNN_HM.hnn_hm.hypergraph import HypergraphsTuple
from methods.HNN_HM.hnn_hm.models_trust import HypergraphModelTrust3
from methods.HNN_HM.hnn_hm.config.artery import ArteryConfig

from utils import artery, artery_eval
from tqdm import tqdm

np.set_printoptions(suppress=True)

def set_gpu_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_optimizer(optimizer_name, lr, momentum):
    assert optimizer_name in ['adam', 'sgd']

    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr, amsgrad=True)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)

    return optimizer


class TrueClassProbabilityLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(TrueClassProbabilityLoss, self).__init__()

    def call(self, y_true, tcp_conf_pred):
        tcp_conf = tcp_conf_pred[:, :1]
        tcp_pred = tcp_conf_pred[:, 1:]
        p_target = tf.gather(tcp_pred, y_true, axis=1, batch_dims=1)
        p_target = tf.reshape(p_target, [-1])
        confidence_loss = tf.losses.mean_squared_error(tcp_conf, p_target)
        return tf.reduce_mean(confidence_loss)


def tcp_loss_fn(y_true, tcp_conf, tcp_pred):
    p_target = tf.gather(tcp_pred, y_true, axis=1, batch_dims=1)
    p_target = tf.reshape(p_target, [-1])
    confidence_loss = tf.losses.mean_squared_error(tcp_conf, p_target)
    return tf.reduce_mean(confidence_loss)



class HGM_Trainer(object):

    def __init__(self, params) -> None:
        self.params = params
        self.rand = np.random.RandomState(seed=params.seed)
        set_random_seed(args.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.params.gpu)
        set_gpu_memory_growth()
        self.__init_database__()
        self.__init_dataloader__()
        self.__init_model__()

    def __init_database__(self):
        dataset, _ = artery._load_graph_in_mem(args.data_file_path)
        self.dataset = dataset


    def __init_model__(self):
        model = HypergraphModelTrust3(num_processing_steps=self.params.num_message_passing, 
                                      node_output_size=16,
                                      num_hidden_layer=self.params.num_layers, 
                                      hidden_layer_size=self.params.latent_size,
                                      node_attention=self.params.node_attention,
                                      edge_attention=self.params.edge_attention)
        self.model = model

    def __init_dataloader__(self):
        problem_generator_test = artery.ArteryProblemGeneratorTest(self.rand, self.dataset, self.dataset, self.params.cache, self.params.cache_path)
        cfg = ArteryConfig(self.params.feature_dim)
        graph_creator = artery.ArteryGraphCreator(cfg)
        dataloader_test = artery.DataLoader(problem_generator_test, graph_creator)
        self.dataloader_test = dataloader_test
        
    def __restore__(self):
        print(f"HGM_Trainer.__restore__: {self.params.exp}")
        self.model.load_weights(f"{self.params.exp}/weights.ckpt")

    def demo(self):
        save_path = "demo_output"
        test_sample_id, template_sample_id = "test1_RAO_CAU", "template1_RAO_CAU"
        g0, g1 = self.dataset[test_sample_id]['g'], self.dataset[template_sample_id]['g']
        n0, n1 = g0.number_of_nodes(), g1.number_of_nodes()

        inputs_te, targets_te, ahg = self.dataloader_test.generate_data_test(test_sample_id, template_sample_id)
        outputs, output_confs, _ = self.model(inputs_te)
        
        outputs = outputs[-1].nodes.numpy().reshape(n0, n1)
        output_confs = output_confs[-1].nodes.numpy().reshape(n0, n1)
        gt = targets_te.nodes.numpy().reshape(n0, n1)
        mappings, total, matched, unmatched, output_hug = artery_eval.compute_accuracy_artery_node2(gt, outputs, self.dataset, ahg, save_path, self.params.plot)

        print(mappings)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--data_file_path', type=str, default="data/artery_with_feature")
    parser.add_argument('--cache_path', type=str, default="cache")
    parser.add_argument('--cache', type=bool, default=True)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--exp', type=str, default="saved_models")
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--cv_max', type=int, default=5)
    parser.add_argument('--template_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--feature_dim', type=int, default=121)

    parser.add_argument('--train', type=str, default="demo")
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--trust_threshold', type=float, default=0.5)

    # model
    parser.add_argument('--batch_size_tr', type=int, default=4)
    parser.add_argument('--node_attention', type=bool, default=True)
    parser.add_argument('--edge_attention', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--batch_size_te', type=int, default=1)
    parser.add_argument('--num_message_passing', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--latent_size', type=int, default=64)

    # training
    parser.add_argument('--n_train_iterations', type=int, default=100001)
    parser.add_argument('--eval_interval', type=int, default=2000)
    parser.add_argument('--plot', type=bool, default=True)

    # optimizer
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--momentum', type=float, default=0.9)

    args = parser.parse_args()

    assert args.train == "demo"
    hgm_trainer = HGM_Trainer(args)
    hgm_trainer.__restore__()
    hgm_trainer.demo()