import argparse
from copy import copy, deepcopy
from collections import defaultdict
from datetime import timedelta
import gc
import gzip
import os
import os.path as osp
import pickle
import psutil
import pdb
import subprocess
import sys
import threading
import time
import traceback
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torch_geometric.nn as gnn
import torch_geometric
import random

from mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended
import model_atom, model_block, model_fingerprint
# from metrics import Evaluator
from metrics import eval_mols
import torch_scatter
import math
from einops import rearrange, reduce, repeat

def set_seed(
    seed: int, deterministic_torch: bool = False
):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    # torch_geometric.seed_everything(seed)
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True)

tmp_dir = "/tmp/molexp"
os.makedirs(tmp_dir, exist_ok=True)

parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", default=1e-4, help="Learning rate", type=float)
parser.add_argument("--mbsize", default=128, help="Minibatch size", type=int)
parser.add_argument("--opt_beta", default=0.9, type=float)
parser.add_argument("--opt_beta2", default=0.999, type=float)
parser.add_argument("--opt_epsilon", default=1e-8, type=float)
parser.add_argument("--nemb", default=256, help="#hidden", type=int)
parser.add_argument("--min_blocks", default=2, type=int)
parser.add_argument("--max_blocks", default=8, type=int)
parser.add_argument("--num_iterations", default=150000, type=int)
parser.add_argument("--num_conv_steps", default=10, type=int)
parser.add_argument("--log_reg_c", default=2.5e-5, type=float)
parser.add_argument("--reward_exp", default=10, type=float)
parser.add_argument("--reward_norm", default=8, type=float)
parser.add_argument("--sample_prob", default=1, type=float)
parser.add_argument("--R_min", default=0.1, type=float)
parser.add_argument("--leaf_coef", default=10, type=float)
parser.add_argument("--clip_grad", default=0, type=float)
parser.add_argument("--clip_loss", default=0, type=float)
parser.add_argument("--replay_mode", default='online', type=str)
parser.add_argument("--bootstrap_tau", default=0, type=float)
parser.add_argument("--weight_decay", default=0, type=float)
parser.add_argument("--random_action_prob", default=0.05, type=float)
parser.add_argument("--array", default='', type=str)
parser.add_argument("--repr_type", default='block_graph')
parser.add_argument("--model_version", default='v4')
parser.add_argument("--run", default=0, help="run", type=int)
parser.add_argument("--save_path", default='results/')
parser.add_argument("--proxy_path", default='./data/pretrained_proxy')
parser.add_argument("--print_array_length", default=False, action='store_true')
parser.add_argument("--progress", default='yes')
parser.add_argument("--floatX", default='float64')
parser.add_argument("--include_nblocks", default=False)
parser.add_argument("--balanced_loss", default=True)
parser.add_argument("--do_wrong_thing", default=False)
parser.add_argument("--data_size", default="medium", type=str)
parser.add_argument("--debug", default=False, action='store_true')
parser.add_argument("--alpha_1", default=0.01, type=float)
parser.add_argument("--alpha_multiplier", default=10.0, type=float)
parser.add_argument("--use_automatic_entropy_tuning", default=1, type=float)
parser.add_argument("--tune_learning_rate", default=2e-5, type=float)
parser.add_argument("--seed", default=1, type=int)

parser.add_argument("--obj", default='qm', type=str)
parser.add_argument("--N", default=16, type=int)
parser.add_argument("--quantile_dim", default=256, type=int)
parser.add_argument("--nvec", default=256, type=int)
parser.add_argument("--ts", default=False, action='store_true')
parser.add_argument("--quick_run", default=False, action='store_true')
parser.add_argument("--load_data", default=False, action='store_true')
parser.add_argument("--mask_path",  default='mask_data/forward_mask.pkl', type=str)
parser.add_argument("--size",  default=200000, type=int)

class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant

class Dataset:

    def __init__(self, args, bpath, device, floatX=torch.double):
        self.test_split_rng = np.random.RandomState(142857)
        self.train_rng = np.random.RandomState(142857)
        self.train_mols = []
        self.test_mols = []
        self.train_mols_map = {}
        self.mdp = MolMDPExtended(bpath)
        self.mdp.post_init(device, args.repr_type, include_nblocks=args.include_nblocks)
        self.mdp.build_translation_table()
        self._device = device
        self.seen_molecules = set()
        self.stop_event = threading.Event()
        self.target_norm = [-8.6, 1.10]
        self.action_num_per_stem = 105
        self.sampling_model = None
        self.sampling_model_prob = 0
        self.floatX = floatX
        self.mdp.floatX = self.floatX
        #######
        # This is the "result", here a list of (reward, BlockMolDataExt, info...) tuples
        self.sampled_mols = []

        get = lambda x, d: getattr(args, x) if hasattr(args, x) else d
        self.min_blocks = get('min_blocks', 2)
        self.max_blocks = get('max_blocks', 10)
        self.mdp._cue_max_blocks = self.max_blocks
        self.replay_mode = get('replay_mode', 'dataset')
        self.reward_exp = get('reward_exp', 1)
        self.reward_norm = get('reward_norm', 1)
        self.random_action_prob = get('random_action_prob', 0)
        self.R_min = get('R_min', 1e-8)
        self.do_wrong_thing = get('do_wrong_thing', False)
        self.data_size=args.data_size
        self.alpha_multiplier =args.alpha_multiplier
        self.use_automatic_entropy_tuning= args.use_automatic_entropy_tuning
        self.quick_run=args.quick_run
        self.args=args

        if self.use_automatic_entropy_tuning>0:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=args.tune_learning_rate,
            )
            print('tune lr:',args.tune_learning_rate)
        else:
            self.log_alpha = Scalar(0.0)
        

        self.online_mols = []
        self.max_online_mols = 1000
        self.offline_mols = []
        self.trans_table = set()
        self.forward_mask = []
        self.end_state = set()
        self.all_mols =set()
        self.load()
        # self.evaluator= Evaluator(self.end_state)
    
    def load(self):
        data_path = f'offline.pkl'
        mask_path = self.args.mask_path
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        with open(mask_path, 'rb') as f:
            self.forward_mask = pickle.load(f)
        self.offline_mols = data
        self.offline_mols = self.offline_mols[:len(self.forward_mask)-1]
        if self.quick_run:
            self.offline_mols = self.offline_mols[:200]
        if self.data_size=="small":
            self.offline_mols = self.offline_mols[:10000]
            self.forward_mask = self.forward_mask[:10000]
        elif self.data_size=="medium":
            self.offline_mols = self.offline_mols[:86000]
            self.forward_mask = self.forward_mask[:86000]
        for i in self.offline_mols:
            self.all_mols.add(i[3].smiles)
        for i in self.offline_mols:
            self.trans_table.add((i[0].smiles,i[3].smiles))
            if i[4]:
                self.end_state.add(i[3].smiles)
        print('load done with data size',len(self.offline_mols))

    def _alpha_and_alpha_loss(self, log_pi: torch.Tensor):
        if self.use_automatic_entropy_tuning>0:
            alpha_loss = -(
                self.log_alpha() * (log_pi - (self.action_num_per_stem) ).detach()
            ).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha , alpha_loss=self.alpha_multiplier , 0
        return alpha, alpha_loss
    
    def process_forward_mask(self):
        with open('mask_data/forward_mask.pkl', 'rb') as f:
            self.forward_mask = pickle.load(f)
        cntt=0
        for trans in self.offline_mols:
            print('process ',cntt)
            cntt+=1
            if cntt<=len(self.forward_mask):
                continue
            p, a, r, s, d = trans
            all_mask=[]
            for stem_idx in range(len(s.stems)):
                mask = [False for _ in range(self.action_num_per_stem)]
                for block_id in range(self.action_num_per_stem):
                    new_mol = self.mdp.add_block_to(s, block_id, stem_idx)
                    if (s.smiles,new_mol.smiles) in self.trans_table:
                        mask[block_id]=True
                all_mask.append(mask)
            self.forward_mask.append(all_mask)
            if cntt%1000==0:
                with open('mask_data/forward_mask.pkl', 'wb') as f:
                    pickle.dump(self.forward_mask, f)
        with open('mask_data/forward_mask.pkl', 'wb') as f:
            pickle.dump(self.forward_mask, f)
        print('process forward mask done.');input()
        

    def _get(self, i, dset):
        if ((self.sampling_model_prob > 0 and # don't sample if we don't have to
             self.train_rng.uniform() < self.sampling_model_prob)
            or len(dset) < 32):
                return self._get_sample_model()
        # Sample trajectories by walking backwards from the molecules in our dataset

        # Handle possible multithreading issues when independent threads
        # add/substract from dset:
        while True:
            try:
                m = dset[i]
            except IndexError:
                i = self.train_rng.randint(0, len(dset))
                continue
            break
        if not isinstance(m, BlockMoleculeDataExtended):
            m = m[-1]
        r = m.reward
        done = 1
        samples = []
        # a sample is a tuple (parents(s), parent actions, reward(s), s, done)
        # an action is (blockidx, stemidx) or (-1, x) for 'stop'
        # so we start with the stop action, unless the molecule is already
        # a "terminal" node (if it has no stems, no actions).
        if len(m.stems):
            samples.append(((m,), ((-1, 0),), r, m, done))
            r = done = 0
        while len(m.blocks): # and go backwards
            parents, actions = zip(*self.mdp.parents(m))
            samples.append((parents, actions, r, m, done))
            r = done = 0
            m = parents[self.train_rng.randint(len(parents))]
        return samples
    
    

    def set_sampling_model(self, model, proxy_reward, sample_prob=0.5):
        self.sampling_model = model
        self.sampling_model_prob = sample_prob
        self.proxy_reward = proxy_reward

    def _get_sample_model(self):
        m = BlockMoleculeDataExtended()
        samples = []
        max_blocks = self.max_blocks
        trajectory_stats = []
        for t in range(max_blocks):
            s = self.mdp.mols2batch([self.mdp.mol2repr(m)])
            s_o, m_o = self.sampling_model(s)
            ## fix from run 330 onwards
            if t < self.min_blocks:
                m_o = m_o * 0 - 1000 # prevent assigning prob to stop
                                     # when we can't stop
            ##
            logits = torch.cat([m_o.reshape(-1), s_o.reshape(-1)])
            cat = torch.distributions.Categorical(
                logits=logits)
            action = cat.sample().item()
            if self.random_action_prob > 0 and self.train_rng.uniform() < self.random_action_prob:
                action = self.train_rng.randint(int(t < self.min_blocks), logits.shape[0])

            q = torch.cat([m_o.reshape(-1), s_o.reshape(-1)])
            trajectory_stats.append((q[action].item(), action, torch.logsumexp(q, 0).item()))
            if t >= self.min_blocks and action == 0:
                r = self._get_reward(m)
                samples.append(((m,), ((-1,0),), r, m, 1))
                break
            else:
                action = max(0, action-1)
                action = (action % self.mdp.num_blocks, action // self.mdp.num_blocks)
                m_old = m
                m = self.mdp.add_block_to(m, *action)
                if len(m.blocks) and not len(m.stems) or t == max_blocks - 1:
                    # can't add anything more to this mol so let's make it
                    # terminal. Note that this node's parent isn't just m,
                    # because this is a sink for all parent transitions
                    r = self._get_reward(m)
                    samples.append((*zip(*self.mdp.parents(m)), r, m, 1))
                    break
                else:
                    samples.append((*zip(*self.mdp.parents(m)), 0, m, 0))
        p = self.mdp.mols2batch([self.mdp.mol2repr(i) for i in samples[-1][0]])
        qp = self.sampling_model(p, None)
        qsa_p = self.sampling_model.index_output_by_action(
            p, qp[0], qp[1][:, 0],
            torch.tensor(samples[-1][1], device=self._device).long())
        inflow = torch.logsumexp(qsa_p.flatten(), 0).item()
        self.sampled_mols.append((r, m, trajectory_stats, inflow))
        if self.replay_mode == 'online' or self.replay_mode == 'prioritized':
            m.reward = r
            self._add_mol_to_online(r, m, inflow)
        return samples

    def _add_mol_to_online(self, r, m, inflow):
        if self.replay_mode == 'online':
            r = r + self.train_rng.normal() * 0.01
            if len(self.online_mols) < self.max_online_mols or r > self.online_mols[0][0]:
                self.online_mols.append((r, m))
            if len(self.online_mols) > self.max_online_mols:
                self.online_mols = sorted(self.online_mols)[max(int(0.05 * self.max_online_mols), 1):]
        elif self.replay_mode == 'prioritized':
            self.online_mols.append((abs(inflow - np.log(r)), m))
            if len(self.online_mols) > self.max_online_mols * 1.1:
                self.online_mols = self.online_mols[-self.max_online_mols:]

    def _get_reward(self, m):
        rdmol = m.mol
        if rdmol is None:
            return self.R_min
        smi = m.smiles
        if smi in self.train_mols_map:
            return self.train_mols_map[smi].reward
        return self.r2r(normscore=self.proxy_reward(m))

    def sample(self, n):
        if self.replay_mode == 'dataset':
            eidx = self.train_rng.randint(0, len(self.train_mols), n)
            samples = sum((self._get(i, self.train_mols) for i in eidx), [])
        elif self.replay_mode == 'online':#tocheck
            eidx = self.train_rng.randint(0, max(1,len(self.online_mols)), n)
            samples = sum((self._get(i, self.online_mols) for i in eidx), [])
        elif self.replay_mode == 'prioritized':
            if not len(self.online_mols):
                # _get will sample from the model
                samples = sum((self._get(0, self.online_mols) for i in range(n)), [])
            else:
                prio = np.float32([i[0] for i in self.online_mols])
                eidx = self.train_rng.choice(len(self.online_mols), n, False, prio/prio.sum())
                samples = sum((self._get(i, self.online_mols) for i in eidx), [])
        return zip(*samples)

    def evaluate(self, epoch,alpha):
        avg_topk_rs, avg_topk_tanimoto, num_modes_above_7_5, num_modes_above_8_0, \
            num_mols_above_7_5, num_mols_above_8_0 = eval_mols(self.sampled_mols,
                reward_norm=args.reward_norm, reward_exp=args.reward_exp, algo="gfn")
        avg_topk_rs_recent, avg_topk_tanimoto_recent, num_modes_above_7_5_recent, num_modes_above_8_0_recent, \
            num_mols_above_7_5_recent, num_mols_above_8_0_recent = eval_mols(self.sampled_mols[-50000:],
                reward_norm=args.reward_norm, reward_exp=args.reward_exp, algo="gfn")
        print(f"state_visited={len(self.sampled_mols)};"
                f"num_modes R>7.5={num_modes_above_7_5};num_mols_above_7_5={num_mols_above_7_5};"
                f"num_modes R>8={num_modes_above_8_0};num_mols_above_8_0={num_mols_above_8_0};reward_top100:{avg_topk_rs[100]}")
        print(f"state_recent;"
                f"num_modes R>7.5 Recent={num_modes_above_7_5_recent};num_mols_above_7_5={num_mols_above_7_5_recent};"
                f"num_modes R>8={num_modes_above_8_0_recent};num_mols_above_8_0={num_mols_above_8_0_recent};reward_top100:{avg_topk_rs_recent[100]}")

    def convert_to_boolean(self,nested_list):
        # 检查是否是列表
        if isinstance(nested_list, list):
            return [self.convert_to_boolean(x) for x in nested_list]
        else:
            # 将0映射为False，将1和2映射为True
            return bool(nested_list)

    def offline_sample(self, n): 
        eidx = self.train_rng.randint(0, max(1,len(self.offline_mols)), n)
        samples = []
        masks = []
        res,resnum,resmx=0,0,0
        for i in eidx:
            a,r,m,done=self.offline_mols[i][1],self.offline_mols[i][2],self.offline_mols[i][3],self.offline_mols[i][4]
            if a != (-1,0):
                pa,pb=self.mdp.parents(m),[]
                for p_a in pa:
                    if (p_a[0].smiles,m.smiles) in self.trans_table:
                        pb.append(p_a)
                    elif p_a[0].smiles in self.all_mols and m.smiles in self.all_mols and self.args.mask_path=='mask_data/new_forward_mask2.pkl':
                        pb.append(p_a)
                samples.append((*zip(*self.mdp.parents(m)),r,m,done, *zip(*pb)  ) )
            else:
                samples.append(((m,), ((-1,0),),r,m,done, (m,), ((-1,0),)  ) )
            masks.extend(self.forward_mask[i])
            if len(self.forward_mask[i])==0:
                masks.append([True for _ in range(self.action_num_per_stem)])#没有stems的在后面会多出一个stem出来，这里要加上全true
        return zip(*samples),masks

    def sample2batch(self, mb):
        p, a, r, s, d, p_b, a_b = mb
        mols = (p, s)
        original_s = s
        # The batch index of each parent
        p_batch = torch.tensor(sum([[i]*len(p) for i,p in enumerate(p)], []),
                               device=self._device).long()
        p_beta_batch = torch.tensor(sum([[i]*len(p_b) for i,p_b in enumerate(p_b)], []),
                               device=self._device).long()
        # Convert all parents and states to repr. Note that this
        # concatenates all the parent lists, which is why we need
        # p_batch
        p = self.mdp.mols2batch(list(map(self.mdp.mol2repr, sum(p, ()))))#先是多个元组，然后合并成一个元组，然后转成repr，然后再转成batch
        p_beta = self.mdp.mols2batch(list(map(self.mdp.mol2repr, sum(p_b, ()))))
        s = self.mdp.mols2batch([self.mdp.mol2repr(i) for i in s])
        # Concatenate all the actions (one per parent per sample)
        a = torch.tensor(sum(a, ()), device=self._device).long()
        a_beta = torch.tensor(sum(a_b, ()), device=self._device).long()
        # rewards and dones
        r = torch.tensor(r, device=self._device).to(self.floatX)
        d = torch.tensor(d, device=self._device).to(self.floatX)
        return (p, p_batch, a, r, s, d, mols, p_beta,p_beta_batch, a_beta,original_s)

    def r2r(self, dockscore=None, normscore=None):
        if dockscore is not None:
            normscore = 4-(min(0, dockscore)-self.target_norm[0])/self.target_norm[1]
        normscore = max(self.R_min, normscore)
        return (normscore/self.reward_norm) ** self.reward_exp


    def start_samplers(self, n, mbsize):
        self.ready_events = [threading.Event() for i in range(n)]
        self.resume_events = [threading.Event() for i in range(n)]
        self.results = [None] * n
        self.results_mask = [None] * n
        def f(idx):
            while not self.stop_event.is_set():
                try:
                    data, data_mask = self.offline_sample(mbsize)
                    self.results[idx] = self.sample2batch(data)
                    self.results_mask[idx]= data_mask
                except Exception as e:
                    print("Exception while sampling:")
                    print(e)
                    self.sampler_threads[idx].failed = True
                    self.sampler_threads[idx].exception = e
                    self.ready_events[idx].set()
                    break
                self.ready_events[idx].set()
                self.resume_events[idx].clear()
                self.resume_events[idx].wait()
        self.sampler_threads = [threading.Thread(target=f, args=(i,)) for i in range(n)]
        [setattr(i, 'failed', False) for i in self.sampler_threads]
        [i.start() for i in self.sampler_threads]
        round_robin_idx = [0]
        def get():
            while True:
                idx = round_robin_idx[0]
                round_robin_idx[0] = (round_robin_idx[0] + 1) % n
                if self.ready_events[idx].is_set():
                    r, msk = self.results[idx], self.results_mask[idx]
                    self.ready_events[idx].clear()
                    self.resume_events[idx].set()
                    return r, msk
                elif round_robin_idx[0] == 0:
                    time.sleep(0.001)
        return get

    def stop_samplers_and_join(self):
        self.stop_event.set()
        if hasattr(self, 'sampler_threads'):
          while any([i.is_alive() for i in self.sampler_threads]):
            [i.set() for i in self.resume_events]
            [i.join(0.05) for i in self.sampler_threads]


def make_model(args, mdp, out_per_mol=1):
    if args.repr_type == 'block_graph':
        if args.obj == "qm":
            model = model_block.DistGraphAgent(nemb=args.nemb,
                                nvec= args.nvec,
                                       out_per_stem=mdp.num_blocks,
                                       out_per_mol=out_per_mol,
                                       num_conv_steps=args.num_conv_steps,
                                       mdp_cfg=mdp,
                                       version=args.model_version,
                                quantile_dim=args.quantile_dim,  
                                n_quantiles=args.N,
                                thompson_sampling=args.ts)
        else:
            model = model_block.GraphAgent(nemb=args.nemb,
                                        nvec=0,
                                        out_per_stem=mdp.num_blocks,
                                        out_per_mol=out_per_mol,
                                        num_conv_steps=args.num_conv_steps,
                                        mdp_cfg=mdp,
                                        version=args.model_version)
    elif args.repr_type == 'atom_graph':
        model = model_atom.MolAC_GCN(nhid=args.nemb,
                                     nvec=0,
                                     num_out_per_stem=mdp.num_blocks,
                                     num_out_per_mol=out_per_mol,
                                     num_conv_steps=args.num_conv_steps,
                                     version=args.model_version,
                                     do_nblocks=(hasattr(args,'include_nblocks')
                                                 and args.include_nblocks), dropout_rate=0.1)
    elif args.repr_type == 'morgan_fingerprint':
        raise ValueError('reimplement me')
        model = model_fingerprint.MFP_MLP(args.nemb, 3, mdp.num_blocks, 1)
    return model


class Proxy:
    def __init__(self, args, bpath, device):
        eargs = pickle.load(gzip.open(f'{args.proxy_path}/info.pkl.gz'))['args']
        params = pickle.load(gzip.open(f'{args.proxy_path}/best_params.pkl.gz'))
        self.mdp = MolMDPExtended(bpath)
        self.mdp.post_init(device, eargs.repr_type)
        self.mdp.floatX = args.floatX
        self.proxy = make_model(eargs, self.mdp)
        for a,b in zip(self.proxy.parameters(), params):
            a.data = torch.tensor(b, dtype=self.mdp.floatX)
        self.proxy.to(device)

    def __call__(self, m):
        m = self.mdp.mols2batch([self.mdp.mol2repr(m)])
        return self.proxy(m, do_stems=False)[1].item()

_stop = [None]


def train_model_with_proxy(args, model, proxy, dataset, num_steps=None, do_save=True):
    debug_no_threads = True
    device = torch.device('cuda')

    if num_steps is None:
        num_steps = args.num_iterations + 1

    tau = args.bootstrap_tau
    if args.bootstrap_tau > 0:
        target_model = deepcopy(model)

    if do_save:
        exp_dir = f'{args.save_path}/{args.obj}_{args.data_size}_{args.alpha_multiplier}_{args.alpha_1}/'
        os.makedirs(exp_dir, exist_ok=True)


    dataset.set_sampling_model(model, proxy, sample_prob=args.sample_prob)

    def save_stuff():
        pickle.dump([i.data.cpu().numpy() for i in model.parameters()],
                    gzip.open(f'{exp_dir}/params.pkl.gz', 'wb'))

        pickle.dump(dataset.sampled_mols,
                    gzip.open(f'{exp_dir}/sampled_mols.pkl.gz', 'wb'))

        pickle.dump({'train_losses': train_losses,
                     'test_losses': test_losses,
                     'test_infos': test_infos,
                     'time_start': time_start,
                     'time_now': time.time(),
                     'args': args,},
                    gzip.open(f'{exp_dir}/info.pkl.gz', 'wb'))

        pickle.dump(train_infos,
                    gzip.open(f'{exp_dir}/train_info.pkl.gz', 'wb'))

    def load_stuff():
        with gzip.open(f'{exp_dir}/params.pkl.gz', 'rb') as f:
            params = pickle.load(f)
        for p, loaded_p in zip(model.parameters(), params):
            p.data = torch.from_numpy(loaded_p).to(p.device)
        with gzip.open(f'{exp_dir}/sampled_mols.pkl.gz', 'rb') as f:
            dataset.sampled_mols = pickle.load(f)
        dataset.sampled_mols = dataset.sampled_mols[:(len(dataset.sampled_mols) // 400) * 400]

    start = 0
    if args.load_data:
        load_stuff()
        start = len(dataset.sampled_mols)/4


    opt = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay,
                           betas=(args.opt_beta, args.opt_beta2),
                           eps=args.opt_epsilon)

    tf = lambda x: torch.tensor(x, device=device).to(args.floatX)
    tint = lambda x: torch.tensor(x, device=device).long()

    mbsize = args.mbsize
    ar = torch.arange(mbsize)

    if not debug_no_threads:
        sampler = dataset.start_samplers(8, mbsize)

    last_losses = []

    def stop_everything():
        print('joining')
        dataset.stop_samplers_and_join()
    _stop[0] = stop_everything

    train_losses = []
    test_losses = []
    test_infos = []
    train_infos = []
    time_start = time.time()
    time_last_check = time.time()

    loginf = 1000 # to prevent nans
    log_reg_c = args.log_reg_c
    clip_loss = tf([args.clip_loss])
    balanced_loss = args.balanced_loss
    do_nblocks_reg = False
    max_blocks = args.max_blocks
    leaf_coef = args.leaf_coef

    for i in range(start,num_steps):
        if not debug_no_threads:
            r, data_mask = sampler()
            for thread in dataset.sampler_threads:
                if thread.failed:
                    stop_everything()
                    pdb.post_mortem(thread.exception.__traceback__)
                    return
            p, pb, a, r, s, d, mols, p_beta, p_beta_batch, a_beta, original_s = r
        else:
            data, data_mask = dataset.offline_sample(mbsize)
            p, pb, a, r, s, d, mols, p_beta, p_beta_batch, a_beta, original_s = dataset.sample2batch(data)
        # Since we sampled 'mbsize' trajectories, we're going to get
        # roughly mbsize * H (H is variable) transitions
        ntransitions = r.shape[0]
        if args.obj=='fm':
            # state outputs
            if tau > 0:
                with torch.no_grad():
                    stem_out_s, mol_out_s = target_model(s, None)
            else:
                stem_out_s, mol_out_s = model(s, None)
            # parents of the state outputs
            stem_out_p, mol_out_p = model(p, None)
            # index parents by their corresponding actions
            qsa_p = model.index_output_by_action(p, stem_out_p, mol_out_p[:, 0], a)
            stem_out_p_beta, mol_out_p_beta = model(p_beta, None)
            qsa_p_beta = model.index_output_by_action(p_beta, stem_out_p_beta, mol_out_p_beta[:, 0], a_beta)

            # then sum the parents' contribution, this is the inflow
            exp_inflow = (torch.zeros((ntransitions,), device=device, dtype=dataset.floatX)
                        .index_add_(0, pb, torch.exp(qsa_p))) # pb is the parents' batch index 
            exp_inflow_beta = (torch.zeros((ntransitions,), device=device, dtype=dataset.floatX)
                        .index_add_(0, p_beta_batch, torch.exp(qsa_p_beta))) # pb is the parents' batch index
            inflow = torch.log(exp_inflow + log_reg_c)
            inflow_beta = torch.log(exp_inflow_beta + log_reg_c)
            # sum the state's Q(s,a), this is the outflow
            exp_outflow = model.sum_output(s, torch.exp(stem_out_s), torch.exp(mol_out_s[:, 0]))
            alpha, alpha_loss = dataset._alpha_and_alpha_loss(stem_out_s)
            def forward_mask():
                mask = data_mask
                out_mask = [original_s[i].smiles in dataset.end_state for i in range(ntransitions)]
                return mask, out_mask
            mask, out_mask = forward_mask()
            mask, out_mask = torch.tensor(mask, dtype=torch.bool, device=device),torch.tensor(out_mask, dtype=torch.bool, device=device)
            exp_outflow_alpha = exp_outflow * (1-d) 
            stem_out_s_beta = stem_out_s.clone()
            with torch.no_grad():
                stem_out_s_beta[~mask] = -loginf

            mol_out_s_beta = mol_out_s[:, 0].clone()
            # mol_out_s_beta = mol_out_s[:, 0]
            with torch.no_grad():
                mol_out_s_beta[~out_mask] = -loginf

            exp_outflow_beta = model.sum_output(s, torch.exp(stem_out_s_beta), torch.exp(mol_out_s_beta))
            exp_outflow_beta =  exp_outflow_beta * (1-d)
            outflow_alpha, outflow_beta = torch.log(exp_outflow_alpha+ log_reg_c), torch.log(exp_outflow_beta + log_reg_c)

            # include reward and done multiplier, then take the log
            # we're guarenteed that r > 0 iff d = 1, so the log always works
            outflow_plus_r = torch.log(log_reg_c + r + exp_outflow * (1-d))
            losses = _losses = (inflow - outflow_plus_r).pow(2) + args.alpha_1*alpha*(inflow - inflow_beta).pow(2) +  alpha*(outflow_alpha - outflow_beta).pow(2)

            if dataset.use_automatic_entropy_tuning>0:
                dataset.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                dataset.alpha_optimizer.step()

            term_loss = (losses * d).sum() / (d.sum() + 1e-20)
            flow_loss = (losses * (1-d)).sum() / ((1-d).sum() + 1e-20)
            if balanced_loss:
                loss = term_loss * leaf_coef + flow_loss
            else:
                loss = losses.mean()
            opt.zero_grad()
            loss.backward(retain_graph=(not i % 50))

            _term_loss = (_losses * d).sum() / (d.sum() + 1e-20)
            _flow_loss = (_losses * (1-d)).sum() / ((1-d).sum() + 1e-20)
            last_losses.append((loss.item(), term_loss.item(), flow_loss.item()))
            train_losses.append((loss.item(), _term_loss.item(), _flow_loss.item(),
                                term_loss.item(), flow_loss.item()))
        elif args.obj=='qm':
            assert ntransitions == int(s.batch.max() + 1) # num of mols "batch size"
            assert tau == 0.

            n_quantile_in = n_quantile_out = args.N
            quantiles = torch.rand((ntransitions, n_quantile_in), dtype=dataset.floatX).to(device)
            # inflow
            in_quantiles = torch.gather(quantiles, 0, repeat(pb, "npar -> npar nq_in", nq_in=n_quantile_in))
            in_quantiles_beta = torch.gather(quantiles, 0, repeat(p_beta_batch, "npar -> npar nq_in", nq_in=n_quantile_in))#
            stem_out_p, mol_out_p = model.forward_with_quantile(p, in_quantiles)
            stem_out_p_beta, mol_out_p_beta = model.forward_with_quantile(p_beta, in_quantiles_beta)#

            qsa_p = model.index_output_by_action(p, stem_out_p, mol_out_p[..., 0], a)
            qsa_p_beta = model.index_output_by_action(p_beta,stem_out_p_beta,mol_out_p_beta[...,0],a_beta)#

            max_qsap = qsa_p.max(dim=0, keepdim=True)[0] # (1, nq)
            max_qsap_beta = qsa_p_beta.max(dim=0, keepdim=True)[0]

            exp_inflow = torch.zeros((ntransitions, n_quantile_in), device=device, dtype=dataset.floatX).\
                index_add_(0, pb, torch.exp(qsa_p - max_qsap)) # pb is the parents' batch index
            inflow = torch.logaddexp(exp_inflow.log(), np.log(log_reg_c) - max_qsap) + max_qsap 

            exp_inflow_beta = torch.zeros((ntransitions, n_quantile_in), device=device, dtype=dataset.floatX).\
                index_add_(0,p_beta_batch,torch.exp(qsa_p_beta-max_qsap_beta))
            inflow_beta = torch.logaddexp(exp_inflow_beta.log(), np.log(log_reg_c) - max_qsap_beta) +max_qsap_beta

            # outflow
            out_quantiles = torch.rand((ntransitions, n_quantile_out), dtype=dataset.floatX).to(device)
            stem_out_s, mol_out_s = model.forward_with_quantile(s, out_quantiles)

            size = int(s.stems_batch.max().item() + 1)
            outflow = torch.logaddexp(
                torch_scatter.scatter_logsumexp(stem_out_s, s.stems_batch, dim=0, dim_size=size).\
                    logsumexp(dim=-1),
                mol_out_s[:, ..., 0]
            ) # same as exp_outflow2.log()

            outflow = torch.logaddexp(outflow, np.log(log_reg_c)*torch.ones_like(outflow)) # care less about tiny flows
            rep_d = repeat(d, "n_tran -> n_tran nq", nq=n_quantile_out)
            rep_r = repeat(r, "n_tran -> n_tran nq", nq=n_quantile_out)
            outflow_plus_r = torch.where(rep_d > 0, (rep_r+log_reg_c).log(), outflow)

            alpha, alpha_loss = dataset._alpha_and_alpha_loss(stem_out_s)
            mask = data_mask
            out_mask = [original_s[i].smiles in dataset.end_state for i in range(ntransitions)]
            # mask, out_mask = forward_mask(data_mask)
            mask, out_mask = torch.tensor(mask, dtype=torch.bool, device=device),torch.tensor(out_mask, dtype=torch.bool, device=device)
            rep_mask = repeat(mask, "n_tran act_num -> n_tran nq act_num",nq=n_quantile_out)
            rep_out_mask = repeat(out_mask, "n_tran -> n_tran nq",nq=n_quantile_out)

            stem_out_s_beta = stem_out_s.clone()
            with torch.no_grad():
                stem_out_s_beta[~rep_mask] = -loginf

            mol_out_s_beta = mol_out_s[:,..., 0].clone()
            with torch.no_grad():
                mol_out_s_beta[~rep_out_mask] = -loginf
            outflow_beta = torch.logaddexp(
                torch_scatter.scatter_logsumexp(stem_out_s_beta, s.stems_batch, dim=0, dim_size=size).\
                    logsumexp(dim=-1),
                mol_out_s_beta
            ) 

            outflow_beta = torch.logaddexp(outflow_beta, np.log(log_reg_c)*torch.ones_like(outflow_beta)) # care less about tiny flows

            diff = repeat(outflow_plus_r, "b nq_out -> b 1 nq_out") - repeat(inflow, "b nq_in -> b nq_in 1")
            abs_weight = torch.abs(repeat(quantiles, "b nq_in -> b nq_in nq_out", nq_out=n_quantile_out) \
                - diff.le(0).float())
            losses = torch.nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none")
            losses = (abs_weight * losses).sum(dim=-2).mean(dim=-1) # sum over qunatile_in, mean over quantile_out

            out_diff = outflow - outflow_beta
            out_diff_losses = torch.nn.functional.smooth_l1_loss(out_diff, torch.zeros_like(out_diff), reduction="none")
            out_diff_losses = out_diff_losses*(1-d.unsqueeze(1))
            out_abs_weight = torch.abs(out_quantiles - out_diff.le(0).float())

            in_diff = inflow - inflow_beta
            in_diff_losses = torch.nn.functional.smooth_l1_loss(in_diff, torch.zeros_like(in_diff), reduction="none")
            in_abs_weight = torch.abs(quantiles - in_diff.le(0).float())
            losses = losses + alpha*(out_abs_weight*out_diff_losses).sum(dim=-1) + args.alpha_1*alpha*(in_abs_weight*in_diff_losses).sum(dim=-1)
            if clip_loss > 0:
                ld = losses.detach()
                losses = losses / ld * torch.minimum(ld, clip_loss)
            term_loss = (losses * d).sum() / (d.sum() + 1e-20)
            flow_loss = (losses * (1-d)).sum() / ((1-d).sum() + 1e-20)
            if balanced_loss:
                loss = term_loss * leaf_coef + flow_loss
            else:
                loss = losses.mean()
            opt.zero_grad()
            loss.backward()
            if dataset.use_automatic_entropy_tuning>0:
                dataset.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                dataset.alpha_optimizer.step()
            last_losses.append((loss.item(), term_loss.item(), flow_loss.item()))
            
        if not i % 50 and args.obj!='qm':
            train_infos.append((
                _term_loss.data.cpu().numpy(),
                _flow_loss.data.cpu().numpy(),
                exp_inflow.data.cpu().numpy(),
                exp_outflow.data.cpu().numpy(),
                r.data.cpu().numpy(),
                mols[1],
                [i.pow(2).sum().item() for i in model.parameters()],
                torch.autograd.grad(loss, qsa_p, retain_graph=True)[0].data.cpu().numpy(),
                torch.autograd.grad(loss, stem_out_s, retain_graph=True)[0].data.cpu().numpy(),
                torch.autograd.grad(loss, stem_out_p, retain_graph=True)[0].data.cpu().numpy(),
            ))
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(),
                                           args.clip_grad)
        opt.step()
        model.training_steps = i + 1
        if tau > 0:
            for _a,b in zip(model.parameters(), target_model.parameters()):
                b.data.mul_(1-tau).add_(tau*_a)

        if not i % 100:
            last_losses = [np.round(np.mean(i), 3) for i in zip(*last_losses)]
            print(i, last_losses,'alpha:', alpha)
            print('time:', time.time() - time_last_check)
            time_last_check = time.time()
            last_losses = []
            dataset.sample(400)
            if not i % 1000 and do_save:
                save_stuff()
        print_interval = 1000 if i<50000 else 5000
        if not i%print_interval:
            dataset.evaluate(i,alpha)
    for i in range(num_steps,1e6):
        dataset.sample(40000)
        dataset.evaluate(i,alpha)
    stop_everything()
    if do_save:
        save_stuff()
    return model

def load():
    with open('offline.pkl', 'rb') as f:
        data = pickle.load(f)
    print( type(data))

def main(args):
    if args.seed!=-1:
        set_seed(args.seed)
        print('seed fixed to',args.seed)
    bpath = "data/blocks_PDB_105.json"
    print('load begin')
    device = torch.device('cuda')

    if args.floatX == 'float32':
        args.floatX = torch.float
    else:
        args.floatX = torch.double
    dataset = Dataset(args, bpath, device, floatX=args.floatX)
    print(args)


    mdp = dataset.mdp

    model = make_model(args, mdp)
    model.to(args.floatX)
    model.to(device)

    proxy = Proxy(args, bpath, device)

    train_model_with_proxy(args, model, proxy, dataset, do_save=True)
    print('Done.')


good_config = {
    'replay_mode': 'online',
    'sample_prob': 1,
    'mbsize': 4,
    'max_blocks': 8,
    'min_blocks': 2,
    # This repr actually is pretty stable
    'repr_type': 'block_graph',
    'model_version': 'v4',
    'nemb': 256,
    # at 30k iterations the models usually have "converged" in the
    # sense that the reward distribution doesn't get better, but the
    # generated molecules keep being unique, so making this higher
    # should simply provide more high-reward states.
    'num_iterations': 30000,

    'R_min': 0.1,
    'log_reg_c': (0.1/8)**4,
    # This is to make reward roughly between 0 and 1 (proxy outputs
    # between ~0 and 10, but very few are above 8).
    'reward_norm': 8,
    # you can play with this, higher is more risky but will give
    # higher rewards on average if it succeeds.
    'reward_exp': 10,
    'learning_rate': 5e-4,
    'num_conv_steps': 10, # More steps is better but more expensive
    # Too low and there is less diversity, too high and the
    # high-reward molecules become so rare the model doesn't learn
    # about them, 0.05 and 0.02 are sensible values
    'random_action_prob': 0.05,
    'opt_beta2': 0.999, # Optimization seems very sensitive to this,
                        # default value works fine
    'leaf_coef': 10, # Can be much bigger, not sure what the trade off
                     # is exactly though
    'include_nblocks': False,
}

if __name__ == '__main__':
  args = parser.parse_args()
  if 0:
    all_hps = eval(args.array)(args)
    for run in range(len(all_hps)):
      args.run = run
      hps = all_hps[run]
      for k,v in hps.items():
        setattr(args, k, v)
      exp_dir = f'{args.save_path}/{args.array}_{args.run}/'
      #if os.path.exists(exp_dir):
      #  continue
      print(hps)
      main(args)
  elif args.array:
    all_hps = eval(args.array)(args)

    if args.print_array_length:
      print(len(all_hps))
    else:
      hps = all_hps[args.run]
      print(hps)
      for k,v in hps.items():
        setattr(args, k, v)
    try:
        main(args)
    except KeyboardInterrupt as e:
        print("stopping for", e)
        _stop[0]()
        raise e
    except Exception as e:
        print("exception", e)
        _stop[0]()
        raise e
  else:
      try:
          main(args)
      except KeyboardInterrupt as e:
          print("stopping for", e)
          _stop[0]()
          raise e
      except Exception as e:
          print("exception", e)
          _stop[0]()
          raise e
