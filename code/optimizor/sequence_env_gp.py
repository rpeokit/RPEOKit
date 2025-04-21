# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import torch
import random
from typing import List, Union
import copy
import pandas as pd

wt_sequence = "MKKYWNRRKNRVEDFINKLTSQLSKLFPDAIFIFEDLDKFNMYDKNSNFNRNLDRTNWRKIAKKLEYKSVVLYVNPHYTSKTCPVCGSKMKSQEGQVVKCDKCGIFDRQFVRCYNIFKRGVELAKKLLGGVGVPVAGAEVDDLLSNEPRGELRLVKPNPNVEAKLPVRKSNRRFELQNPKDFVQIFDFPLMVYTVDLNGKYLKIYNCP"

##start_seqs
df_seq_1 = pd.read_csv(r"./input_file/tnpb_model_input_data_new.csv")
initi_seqs = list(df_seq_1["seqs"].values)

#map dict
ind_arr_list = list(np.load(r"./input_file/tnpb_pos_1_new.npy")[0])
def pre_select_dict(pre_select_list):
    d_dict = {}
    mutate_str_list = []
    for i in range(len(pre_select_list)):
        d_dict[i] = pre_select_list[i]
    return d_dict #, mutate_str
map_dict = pre_select_dict(ind_arr_list)
av_df = pd.read_csv(r"./input_file/tnpb_pos_space_new_wt.csv")


colums = list(av_df.columns)[1:]

av_arr = av_df[colums[0]].values
for i in range(1,len(colums)):
    av_arr = np.concatenate([av_arr,av_df[colums[i]].values],axis=0)
re_moves = list(np.where(av_arr==0)[0])

#print(list(re_moves[0]))
######move#######
#
AAS = "ILVAGMFYWEDQNHCRKSTP"
def feature_single(variant):
    Feature = []
    aalist = list(AAS)
    for AA in variant:
        Feature.append([AA == aa for aa in aalist])
    Feature = np.asarray(Feature).astype(float)
    if len(Feature.shape) == 2:
        features = np.reshape(Feature, [Feature.shape[0] * Feature.shape[1]])

    return features
def string_to_one_hot(sequence: str, alphabet: str) -> np.ndarray:

    out = np.zeros((len(sequence), len(alphabet)))
    for i in range(len(sequence)):
        out[i, alphabet.index(sequence[i])] = 1
    return out

def one_hot_to_string(
    one_hot: Union[List[List[int]], np.ndarray], alphabet: str
) -> str:
    """
    Return the sequence string representing a one-hot vector according to an alphabet.

    Args:
        one_hot: One-hot of shape `(len(sequence), len(alphabet)` representing
            a sequence.
        alphabet: Alphabet string (assigns each character an index).

    Returns:
        Sequence string representation of `one_hot`.

    """
    residue_idxs = np.argmax(one_hot, axis=1)
    return "".join([alphabet[idx] for idx in residue_idxs])
# 将截短的变异序列映射回全长序列
def map_to_full_len_gfp(start_seq, d_dict, truncated_str):
    wt_list = list(start_seq)
    for k, v in d_dict.items():
        wt_list[v] = truncated_str[k]
    gfp_mutate_seq = "".join(wt_list)
    return gfp_mutate_seq

def string_to_feature(string):
    seq_list = []
    seq_list.append(string)
    seq_np = np.array(
        [string_to_one_hot(seq, AAS) for seq in seq_list]
    )
    one_hots = torch.from_numpy(seq_np)
    one_hots = one_hots.to(torch.float32)
    return one_hots

class Seq_env(object):
    """sequence space for the env"""
    def __init__(self,
                 seq_len,
                 alphabet,
                 model,
                 starting_seq_pool,
                 trust_radus,
                 ):

        self.move_count = 0

        self.seq_len = seq_len
        self.vocab_size = len(alphabet)

        self.alphabet = alphabet
        self.model = model

        self.start_seq_pool = initi_seqs

        starting_seq = self.start_seq_pool[0]
        self.starting_seq = starting_seq
        self.seq = starting_seq
        self.start_seq_pool.remove(starting_seq)

        self.init_state = string_to_one_hot(self.seq, self.alphabet).astype(np.float32)
        self.previous_init_state = string_to_one_hot(self.seq, self.alphabet).astype(np.float32)
        self.init_state_count = 0

        self.unuseful_move = 0
        self.states = {}
        
        self.repeated_seq_ocurr = False
        self.episode_seqs = starting_seq_pool
        ###debug
        self.playout = 0
        ###debug

    def init_seq_state(self): #start_player=0


        self.previous_fitness = -float("inf")
        self.move_count = 0

        self.unuseful_move = 0

        self.repeated_seq_ocurr = False

        self._state = copy.deepcopy(self.init_state)
            
        combo = one_hot_to_string(self._state, AAS)

        self.init_combo = combo

        feature = []
        
        input = feature_single(combo)
        input = np.expand_dims(input, axis=0)
        outputs = self.model.predict(input)[0]
        
        self._state_fitness = outputs
    

        self.availables = list(range(self.seq_len * self.vocab_size))
        for remove in re_moves:
            self.availables.remove(remove)

        self.states = {}
        self.last_move = -1

        self.previous_init_state = copy.deepcopy(self._state)


    def current_state(self):

        square_state = np.zeros((self.seq_len, self.vocab_size))
        square_state = self._state
        return square_state.T
    def do_mutate(self, move):
        
        self.previous_fitness = self._state_fitness
        self.move_count += 1
        self.availables.remove(move)
        pos = move // self.vocab_size
        res = move % self.vocab_size

        if self._state[pos, res] == 1:
            self.unuseful_move = 1
            self._state_fitness = 0.0
        else:
            self._state[pos] = 0
            self._state[pos, res] = 1

            
            combo = one_hot_to_string(self._state, AAS)
            
            input = feature_single(combo)
            input = np.expand_dims(input, axis=0)
            outputs = self.model.predict(input)[0]
            

            self._state_fitness = outputs
        
        current_seq = one_hot_to_string(self._state, AAS)

        if self._state_fitness >0.6 and self.playout==0 :
            print("hehe")
        if current_seq in self.episode_seqs:
            self.repeated_seq_ocurr = True
            self._state_fitness = 0.0
            if self.playout==0:
                print("")
            
        else:
            self.episode_seqs.append(current_seq)

        if self._state_fitness > self.previous_fitness:  # 0.6* 0.75*   #and not repeated_seq_ocurr
            
            self.init_state = copy.deepcopy(self._state)
            self.init_state_count = 0


        self.last_move = move




    def mutation_end(self):
        
        if self.repeated_seq_ocurr == True:
            return True
       
        if self.unuseful_move == 1:
            return True
        if self._state_fitness < self.previous_fitness:  # 0.6* 0.75*

            return True

        return False

class Mutate(object):
    """mutating server"""

    def __init__(self, Seq_env, **kwargs):
        self.Seq_env = Seq_env

    def start_p_mutating(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.Seq_env.init_board(start_player)
        p1, p2 = self.Seq_env.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.Seq_env, player1.player, player2.player)
        while True:
            current_player = self.Seq_env.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.Seq_env)
            self.Seq_env.do_move(move)
            if is_shown:
                self.graphic(self.Seq_env, player1.player, player2.player)
            end, winner = self.Seq_env.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_mutating(self, mutater, is_shown=0, temp=1e-3):
        """ start mutating using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        #
        if (self.Seq_env.previous_init_state == self.Seq_env.init_state).all():
            self.Seq_env.init_state_count += 1
        if self.Seq_env.init_state_count >= 10:  #10,6,7,5,8

            new_start_seq = self.Seq_env.start_seq_pool[0]
            self.Seq_env.init_state = string_to_one_hot(new_start_seq, self.Seq_env.alphabet).astype(np.float32)
            self.Seq_env.start_seq_pool.remove(new_start_seq)  
            self.Seq_env.init_state_count = 0
        #
        self.Seq_env.init_seq_state()
        print("起始序列：{}".format(self.Seq_env.init_combo))

        states, mcts_probs, reward_z = [], [], []

        generated_seqs, seq_fitness,fragment_seqs = [], [], []
        while True:
            move, move_probs = mutater.get_action(self.Seq_env,
                                                 temp=temp,
                                                 return_prob=1)
            if move:

                states.append(self.Seq_env.current_state())
                mcts_probs.append(move_probs)
                reward_z.append(self.Seq_env._state_fitness)




                self.Seq_env.do_mutate(move)
                print("move_fitness: %.16f\n" % (self.Seq_env._state_fitness+0.5))
                state_string = one_hot_to_string(self.Seq_env._state, AAS)
                print(state_string)

                combo_str = one_hot_to_string(self.Seq_env._state, AAS)
                full_len_mutation_seq = map_to_full_len_gfp(wt_sequence, map_dict, combo_str)
                generated_seqs.append(full_len_mutation_seq)
                seq_fitness.append(float(self.Seq_env._state_fitness))
                fragment_seqs.append(combo_str)


            end = self.Seq_env.mutation_end()
            if end:
                
                combo_str = one_hot_to_string(self.Seq_env._state, AAS)
                full_len_mutation_seq = map_to_full_len_gfp(wt_sequence, map_dict, combo_str)
                generated_seqs.append(full_len_mutation_seq)
                seq_fitness.append(float(self.Seq_env._state_fitness))
                fragment_seqs.append(combo_str)

                mutater.reset_Mutater()
                if is_shown:

                    print("Mutation end.")
                return zip(states, mcts_probs, reward_z), zip(generated_seqs, seq_fitness, fragment_seqs)#generated_seqs#winner,