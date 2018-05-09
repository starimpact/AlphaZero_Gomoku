# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet with Keras
Tested under Keras 2.0.5 with tensorflow-gpu 1.2.1 as backend

@author: Mingxu Zhang
""" 
import os
os.path.insert(0, '/home/mingzhang/work/dmlc/python_mxnet/python')

from __future__ import print_function
import mxnet as mx

import numpy as np
import pickle


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height, model_file=None):
        self.context = mx.gpu(0)
        self.batchsize = 16
        self.board_width = board_width
        self.board_height = board_height 
        self.l2_const = 1e-4  # coef of l2 penalty 
        self.create_policy_value_net()   

        if model_file:
            net_params = pickle.load(open(model_file, 'rb'))
            self.model.set_weights(net_params)

    def conv_act(self, data, num_filter=32, kernel=(3, 3), act='relu'):
        pad = (int(kernel(0)/2), int(kernel(1)/2))
        conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, pad=pad)
        act1 = conv1
        if act is not None or act!='':
          act1 = mx.sym.Activation(data=conv1, act_type=act)

        return act1

    def create_policy_value_net(self):
        """create the policy value network """   
        input_states_shape = (self.batchsize, 4, self.board_height, self.board_width)
        input_states = mx.sym.Variable(name='input_states', shape=input_states_shape)
        
        conv1 = self.conv_act(input_states, 32, (3, 3))
        conv2 = self.conv_act(conv1, 64, (3, 3))
        conv3 = self.conv_act(conv2, 128, (3, 3))
        # action policy layers
        conv3_1 = self.conv_act(conv3, 4, (1, 1))
        gpool_1 = mx.sym.Pooling(conv3_1, global_pool=True, pool_type='avg')
        fc_1 = mx.sym.FullyConnected(gpool_1, num_hidden = self.board_width*self.board_height)
        action_1 = mx.sym.SoftmaxActivation(fc_1)
        mcts_probs_shape = (self.batchsize, self.board_height * self.board_width)
        mcts_probs = mx.sym.Variable(name='mcts_probs', shape=mcts_probs_shape)
        policy_loss = -mx.sym.sum(mx.sym.log(action_1) * mcts_probs)

        # state value layers
        conv3_2 = self.conv_act(conv3, 2, (1, 1))
        gpool_2 = mx.sym.Pooling(conv3_2, global_pool=True, pool_type='avg')
        fc_2 = mx.sym.FullyConnected(gpool_2, num_hidden=64)
        act2 = mx.sym.Activation(data=fc_2, act_type='relu')
        fc_3 = mx.sym.FullyConnected(act2, num_hidden=1)
        evaluation = mx.sym.Activation(evaluation, act_type='tanh')
        input_labels_shape = (self.batchsize, 1)
        input_labels = mx.sym.Variable(name='input_labels', shape=input_labels_shape)
        value_loss = mx.sym.mean(mx.sym.square(input_labels - evaluation))

        policy_value_loss_group = mx.sym.Group([policy_loss, value_loss])
        policy_value_loss = mx.sym.MakeLoss(policy_value_loss_group) 
        
        policy_value_output = mx.sym.Group([action_1, evaluation])

        self.pv_train = mx.mod.Module(symbol=policy_value_loss, 
                                           data_names=['input_states'],
                                           label_names=['input_labels', 'mcts_probs'],
                                           context=self.context) 
        self.pv_train.bind(data_shapes=[('input_states', input_states)], 
                                                label_shapes=[('input_labels', input_labels_shape, 'mcts_probs', mcts_probs_shape)],
                                                for_training=True)
        self.pv_train.init_params(initializer=mx.init.Xavier())
        self.pv_train.init_optimizer(optimizer='adam', optimizer_params=('learning_rate':0.0002))

        self.pv_predict = mx.mod.Module(symbol=policy_value_output, 
                                           data_names=['input_states'],
                                           label_names=None,
                                           context=self.context) 
        
        self.pv_predict.bind(data_shapes=[('input_states', input_states)], for_training=False)
        args, auxs = self.pv_train.get_params()
        self.pv_predict.set_params(args, auxs)
        
    def policy_value(self, state_batch):
        self.pv_predict.forward(mx.io.DataBatch([state_batch], []))
        acts, vals = self.pv_predict.get_outputs()
        return acts, vals
       
    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        legal_positions = board.availables
        current_state = board.current_state().reshape(self.batchsize, 4, self.board_height, self.board_width)
        current_state = mx.nd.array(current_state)

        act_probs, values = self.pv_predict.policy_value(current_state)
        act_probs = act_probs.asnumpy()
        values = values.asnumpy()
        return act_probs, values

    def train_step(self, state_batch, mcts_probs, winner_batch, learning_rate):
        self.pv_train.forward(mx.io.DataBatch([state_batch], [mcts_probs, winner_batch]))
        self.pv_train.backward()
        self.pv_train.update()
        args, auxs = self.pv_train.get_params()
        self.pv_predict.set_params(args, auxs)

    def get_policy_param(self):
        net_params = self.pv_train.get_params()        
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()
        pickle.dump(net_params, open(model_file, 'wb'), protocol=2)
