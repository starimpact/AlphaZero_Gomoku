# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import pickle
import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_mxnet import PolicyValueNet, PolicyValueNet_SelfPlay  # Keras
import logging
import mpi4py.MPI as MPI
import time

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


logging.basicConfig(filename='example.log',level=logging.DEBUG)

print('I am process ', comm_rank, comm_size)
#@ray.remote(num_gpus=1)
#class Actor(threading.Thread):
class Actor(object):
    def __init__(self, nameid='', gpuid=0, infos=None, init_model=None):
        #super(Actor, self).__init__(name=nameid)
        '''
        infos:(board_height, board_width)
        '''
        print('actor ', comm_rank)

        self.nameid = nameid
        self.gpuid = gpuid
        self.board_height = infos[0]
        self.board_width = infos[1]
        self.n_in_row = infos[2]
        self.temp = infos[3]
        self.c_puct = infos[4]
        self.n_playout = infos[5]
        self.game = Game(Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row))
        if init_model:
            # start training from an initial policy-value net
            self.selfplay_net = PolicyValueNet_SelfPlay(self.board_width,
                                                   self.board_height,
                                                   model_params=init_model)
        else:
            # start training from a new policy-value net
            self.selfplay_net = PolicyValueNet_SelfPlay(self.board_width,
                                                   self.board_height)
        self.selfplay_net.set_params(init_model)
        self.mcts_player = MCTSPlayer(self.selfplay_net.policy_value_fn,
                                  c_puct=self.c_puct,
                                  n_playout=self.n_playout,
                                  is_selfplay=1)

    def Set_Params(self, params):
        self.selfplay_net.set_params(params)

    def Play(self):
        winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
        self.playdata = list(play_data)[:]
        return self.playdata


class TrainPipeline():
    def __init__(self, init_model=None):
        # params of the board and the game
        self.parallel_games = 1
        #self.pool = Pool()
        self.board_width = 8
        self.board_height = 8
        self.n_in_row = 5
        # training params
        self.learn_rate = 2e-4
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 1100
        self.batch_size = 1024  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 1000
        self.game_batch_num = 150000
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000

        self.policy_value_net = None

        if comm_rank == 0:
            if init_model:
                # start training from an initial policy-value net
                self.policy_value_net = PolicyValueNet(self.board_width,
                                                       self.board_height,
                                                       model_params=init_model)
            else:
                # start training from a new policy-value net
                self.policy_value_net = PolicyValueNet(self.board_width,
                                                       self.board_height)

        self.mcts_player = None
        params = None
        infos = (self.board_height, self.board_width, self.n_in_row, self.temp, self.c_puct, self.n_playout)

        if comm_rank>0:
            print('rank ', comm_rank, ' before recv ')
            params = comm.recv(source=0)
            print('rank ', comm_rank, ' after recv ')
            #self.mcts_player = Actor('gamer_'+str(comm_rank), 2, infos, params)

        if self.policy_value_net and comm_rank==0:
            params = self.policy_value_net.get_policy_param()
            print('rank ', comm_rank, ' before bcast')
            #comm.bcast('params', root=0)
            for pi in range(1, comm_size):
                comm.send(params, dest=pi)
            print('rank ', comm_rank, ' after bcast')
            #self.mcts_player = Actor('gamer_'+str(comm_rank), 2, infos, params)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self):
        """collect self-play data for training"""
        datas = self.mcts_player.Play()
        self.episode_len = 0
        for pgi in range(self.parallel_games):
            play_data = datas[pgi]
            _len = len(play_data)
            self.episode_len += _len
            print('game ', pgi, _len)
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)
 
    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        learn_rate = self.learn_rate*self.lr_multiplier
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    learn_rate)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                print('early stopping:', i, self.epochs)
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.05:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 20:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.4f},"
               "lr:{:.1e},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        learn_rate,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data()
                print("batch i:{}, episode_len:{}, buffer_len:{}".format(
                        i+1, self.episode_len, len(self.data_buffer)))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    params = self.policy_value_net.get_policy_param()
                    for spi in range(self.parallel_games):
                        gamer = self.mcts_players[spi]
                        #gamer.Set_Params.remote(params)
                        gamer.Set_Params(params)
                        
                # check the performance of the current model,
                # and save the model params
                if (i+1) % 50 == 0:
                    self.policy_value_net.save_model('./current_policy.model')
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    #ray.init(num_gpus=4)
    model_file = None #'current_policy.model'
    policy_param = None 
    if model_file is not None:
        print('loading...', model_file)
        try:
            policy_param = pickle.load(open(model_file, 'rb'))
        except:
            policy_param = pickle.load(open(model_file, 'rb'),
                                       encoding='bytes')  # To support python3
    training_pipeline = TrainPipeline(policy_param)
    #training_pipeline.run()
    print('rank ', comm_rank, ' bye bye...')

