import argparse
import gzip
import pickle
import itertools
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm

from lib.acquisition_fn import get_acq_fn
from lib.dataset import get_dataset
from lib.generator import get_generator
from lib.logging import get_logger
from lib.oracle_wrapper import get_oracle
from lib.proxy import get_proxy_model
from lib.utils.distance import is_similar, edit_dist
from lib.utils.env import get_tokenizer,Vocab
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", default='results/test_mlp.pkl.gz')
parser.add_argument("--tb_log_dir", default='results/test_mlp')
parser.add_argument("--name", default='test_mlp')
parser.add_argument("--load_scores_path", default='.')

# Multi-round
parser.add_argument("--num_rounds", default=1, type=int)
parser.add_argument("--task", default="amp", type=str)
parser.add_argument("--num_sampled_per_round", default=256*4, type=int) # 10k
parser.add_argument("--num_folds", default=5)
parser.add_argument("--vocab_size", default=21)
parser.add_argument("--max_len", default=65)
parser.add_argument("--gen_max_len", default=50+1)
parser.add_argument("--proxy_uncertainty", default="dropout")
parser.add_argument("--save_scores_path", default=".")
parser.add_argument("--save_scores", action="store_true")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--run", default=-1, type=int)
parser.add_argument("--noise_params", action="store_true")
parser.add_argument("--enable_tensorboard", action="store_true")
parser.add_argument("--save_proxy_weights", action="store_true")
parser.add_argument("--use_uncertainty", action="store_true")
parser.add_argument("--filter", action="store_true")
parser.add_argument("--kappa", default=0.1, type=float)
parser.add_argument("--acq_fn", default="none", type=str)
parser.add_argument("--load_proxy_weights", type=str)
parser.add_argument("--max_percentile", default=80, type=int)
parser.add_argument("--filter_threshold", default=0.1, type=float)
parser.add_argument("--filter_distance_type", default="edit", type=str)
parser.add_argument("--oracle_split", default="D2_target", type=str)
parser.add_argument("--proxy_data_split", default="D1", type=str)
parser.add_argument("--oracle_type", default="MLP", type=str)
parser.add_argument("--oracle_features", default="AlBert", type=str)
parser.add_argument("--medoid_oracle_dist", default="edit", type=str)
parser.add_argument("--medoid_oracle_norm", default=1, type=int)
parser.add_argument("--medoid_oracle_exp_constant", default=6, type=int)


# Generator
parser.add_argument("--gen_learning_rate", default=5e-4, type=float)
parser.add_argument("--gen_Z_learning_rate", default=5e-3, type=float)
parser.add_argument("--gen_clip", default=10, type=float)
parser.add_argument("--gen_num_iterations", default=20000, type=int) # Maybe this is too low?
parser.add_argument("--gen_episodes_per_step", default=16, type=int)
parser.add_argument("--gen_num_hidden", default=128, type=int)
parser.add_argument("--gen_reward_norm", default=1, type=float)
parser.add_argument("--gen_reward_exp", default=8, type=float)
parser.add_argument("--gen_reward_min", default=-8, type=float)
parser.add_argument("--gen_L2", default=0, type=float)
parser.add_argument("--gen_partition_init", default=50, type=float)

# Soft-QLearning/GFlownet gen
parser.add_argument("--gen_reward_exp_ramping", default=1, type=float)
parser.add_argument("--gen_balanced_loss", default=1, type=float)
parser.add_argument("--gen_output_coef", default=10, type=float)
parser.add_argument("--gen_loss_eps", default=1e-5, type=float)
parser.add_argument("--gen_random_action_prob", default=0.01, type=float)
parser.add_argument("--gen_sampling_temperature", default=1., type=float)
parser.add_argument("--gen_leaf_coef", default=25, type=float)
parser.add_argument("--gen_data_sample_per_step", default=16, type=int)
# PG gen
parser.add_argument("--gen_do_pg", default=0, type=int)
parser.add_argument("--gen_pg_entropy_coef", default=1e-2, type=float)
# learning partition Z explicitly
parser.add_argument("--gen_do_explicit_Z", default=0, type=int)
parser.add_argument("--gen_model_type", default="mlp")

# Proxy
parser.add_argument("--proxy_learning_rate", default=1e-4)
parser.add_argument("--proxy_type", default="regression")
parser.add_argument("--proxy_arch", default="mlp")
parser.add_argument("--proxy_num_layers", default=4)
parser.add_argument("--proxy_dropout", default=0.1)

parser.add_argument("--proxy_num_hid", default=64, type=int)
parser.add_argument("--proxy_L2", default=1e-4, type=float)
parser.add_argument("--proxy_num_per_minibatch", default=256, type=int)
parser.add_argument("--proxy_early_stop_tol", default=5, type=int)
parser.add_argument("--proxy_early_stop_to_best_params", default=0, type=int)
parser.add_argument("--proxy_num_iterations", default=30000, type=int)
parser.add_argument("--proxy_num_dropout_samples", default=25, type=int)

parser.add_argument("--enable_offline", default=0, type=int)
parser.add_argument("--data_path", default='results/data_25.0.pkl')
# parser.add_argument("--data_path", default='offline_data_final.pkl')
parser.add_argument("--used_data_path", default='offline_data_final.pkl')

args = parser.parse_args()

with open(args.data_path,'rb') as f:
    x,y = pickle.load(f)
sorted_pairs = sorted(zip(y, x), reverse=True)
y_sorted, x_sorted = zip(*sorted_pairs)

with open(args.used_data_path,'rb') as f:
    xu,yu = pickle.load(f)

sorted_pairs = sorted(zip(yu, xu), reverse=True)
yu, xu = zip(*sorted_pairs)
def mean_pairwise_distances(seqs):
    dists = []
    for pair in itertools.combinations(seqs, 2):
        dists.append(edit_dist(*pair))
        x1,x2 = pair
        assert x1!=x2
    return np.mean(dists)

def calc_diversity(seqs):
    res = 0.0
    # for pair in itertools.combinations(seqs, 2):
    #     res+=edit_dist(*pair)
    for i in seqs:
        for j in seqs:
            res+=edit_dist(i,j)
    return res/(len(seqs)*(len(seqs)-1.0))
def calc_noverty(seqs,d0):
    res = 0
    for i in seqs:
        mindis=0x3f3f3f3f
        for j in d0:
            mindis = min(mindis, edit_dist(i,j))
        res+=mindis
    return res/len(seqs)
    
def top_k_scores(x,y,k,d0):
    res = {}
    res['performance']=np.mean(y[:k])
    #calc diversity
    res['diversity']=calc_diversity(x[:k])
    res['noverty']=calc_noverty(x[:k],d0)
    return res



torch.manual_seed(args.seed)
np.random.seed(args.seed)
args.logger = get_logger(args)
args.device = torch.device('cuda')
oracle = get_oracle(args)

from clamp_common_eval.defaults import get_default_data_splits
source = get_default_data_splits(setting='Target')
data = source.sample(args.proxy_data_split, -1)
# print(len(data['AMP']),len(data['nonAMP']))
oracle_data = data['AMP']
# dataset = get_dataset(args, get_oracle(args))
print(top_k_scores(x_sorted,y_sorted,100,oracle_data))
print(mean_pairwise_distances(x_sorted))  
print(top_k_scores(xu,yu,100,oracle_data)) 
print(len(x_sorted))
all_chars_data = ''.join(x_sorted).replace('%', '')
char_counts = Counter(all_chars_data)
total_chars = sum(char_counts.values())
char_frequencies = {char: count / total_chars for char, count in char_counts.items()}
# 按字母顺序排序
sorted_frequencies1 = dict(sorted(char_frequencies.items()))

# 输出每个字母的频率（按字母顺序）
print(sorted_frequencies1)

# all_chars_data = ''.join(data['AMP']).replace('%', '')
all_chars_data = ''.join(xu).replace('%', '')
char_counts = Counter(all_chars_data)
total_chars = sum(char_counts.values())
char_frequencies = {char: count / total_chars for char, count in char_counts.items()}
# 按字母顺序排序
sorted_frequencies = dict(sorted(char_frequencies.items()))

# 输出每个字母的频率（按字母顺序）
print(sorted_frequencies)

import math
def kl_divergence(p, q):
    # 确保两个字典有相同的字母集
    all_keys = set(p.keys()).union(set(q.keys()))
    # 计算KL散度
    divergence = 0
    for key in all_keys:
        p_value = p.get(key, 0)  # 如果 key 在 p 中不存在，使用 0
        q_value = q.get(key, 0)  # 如果 key 在 q 中不存在，使用 0
        
        # 为了避免出现log(0)，我们可以跳过 q_value 为 0 的情况
        if p_value > 0 and q_value > 0:
            divergence += p_value * math.log(p_value / q_value)
    return divergence

print(kl_divergence(sorted_frequencies1,sorted_frequencies))

import json
with open("distrib_dataset.json", "w") as outfile:
    json.dump(sorted_frequencies, outfile)