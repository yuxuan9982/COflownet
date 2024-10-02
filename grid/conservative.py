import argparse
import copy
import gzip
import heapq
import itertools
import os
import pickle
from collections import defaultdict
from itertools import count

import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import random


parser = argparse.ArgumentParser()

parser.add_argument("--save_path", default='results', type=str)
parser.add_argument("--device", default='cpu', type=str)
parser.add_argument("--progress", action='store_true')

#
parser.add_argument("--method", default='flownet', type=str)
parser.add_argument("--learning_rate", default=1e-5, help="Learning rate", type=float)
parser.add_argument("--opt", default='adam', type=str)
parser.add_argument("--adam_beta1", default=0.9, type=float)
parser.add_argument("--adam_beta2", default=0.999, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--mbsize", default=16, help="Minibatch size", type=int)
parser.add_argument("--train_to_sample_ratio", default=1, type=float)
parser.add_argument("--n_hid", default=256, type=int)
parser.add_argument("--n_layers", default=2, type=int)
parser.add_argument("--n_train_steps", default=40000, type=int)
parser.add_argument("--num_empirical_loss", default=200000, type=int,
                    help="Number of samples used to compute the empirical distribution loss")
# Env
parser.add_argument('--func', default='corners')
parser.add_argument("--horizon", default=8, type=int)
parser.add_argument("--ndim", default=4, type=int)

# MCMC
parser.add_argument("--bufsize", default=16, help="MCMC buffer size", type=int)

# Flownet
parser.add_argument("--bootstrap_tau", default=0., type=float)
parser.add_argument("--replay_strategy", default='none', type=str) # top_k none
parser.add_argument("--replay_sample_size", default=2, type=int)
parser.add_argument("--replay_buf_size", default=100, type=float)

# PPO
parser.add_argument("--ppo_num_epochs", default=32, type=int) # number of SGD steps per epoch
parser.add_argument("--ppo_epoch_size", default=16, type=int) # number of sampled minibatches per epoch
parser.add_argument("--ppo_clip", default=0.2, type=float)
parser.add_argument("--ppo_entropy_coef", default=1e-1, type=float)
parser.add_argument("--clip_grad_norm", default=0., type=float)

# SAC
parser.add_argument("--sac_alpha", default=0.98*np.log(1/3), type=float)

# CQL
parser.add_argument("--dataType", default='null.pt', type=str)
parser.add_argument("--use_automatic_entropy_tuning", default=1, type=float)
parser.add_argument("--alpha_multiplier", default=1.0, type=float)


_dev = [torch.device('cpu')]
tf = lambda x: torch.FloatTensor(x).to(_dev[0])
tl = lambda x: torch.LongTensor(x).to(_dev[0])

def set_device(dev):
    _dev[0] = dev

def func_corners(x):
    ax = abs(x)
    return (ax > 0.5).prod(-1) * 0.5 + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2 + 1e-5

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

class GridEnv:

    def __init__(self, horizon, ndim=2, xrange=[-1, 1], func=None, allow_backward=False):
        self.horizon = horizon
        self.start = [xrange[0]] * ndim
        self.ndim = ndim
        self.width = xrange[1] - xrange[0]
        self.func = (
            (lambda x: ((np.cos(x * 50) + 1) * norm.pdf(x * 5)).prod(-1) + 0.01)
            if func is None else func)
        self.xspace = np.linspace(*xrange, horizon)
        self.allow_backward = allow_backward  # If true then this is a
                                              # MCMC ergodic env,
                                              # otherwise a DAG
        self._true_density = None


    def obs(self, s=None):
        s = np.int32(self._state if s is None else s)
        z = np.zeros((self.horizon * self.ndim), dtype=np.float32)
        # print('s is',s, len(s))
        z[np.arange(len(s)) * self.horizon + s] = 1
        return z
    
    def obs2s(self, obs):
        return np.argmax(obs.reshape((self.ndim, self.horizon)), 1)

    def s2x(self, s):
        return (self.obs(s).reshape((self.ndim, self.horizon)) * self.xspace[None, :]).sum(1)
    
    # [1, 6, 3] -> [1, -1, 0]
    # 1 or -1 means in mode, 0 means not in mode
    def s2mode(self, s):
        ret = np.int32([0] * self.ndim)
        x = self.s2x(s)
        ret += np.int32((x > 0.6) * (x < 0.8))
        ret += -1 * np.int32((x > -0.8) * (x < -0.6))
        return ret

    def reset(self):
        self._state = np.int32([0] * self.ndim) 
        self._step = 0
        return self.obs(), self.func(self.s2x(self._state)), self._state

    def parent_transitions(self, s, used_stop_action):
        if used_stop_action:
            return [self.obs(s)], [self.ndim]
        parents = []
        actions = []
        for i in range(self.ndim):
            if s[i] > 0:
                sp = s + 0
                sp[i] -= 1
                if sp.max() == self.horizon-1: # can't have a terminal parent
                    continue
                parents += [self.obs(sp)]
                actions += [i]
        return parents, actions

    def step(self, a, s=None):
        if self.allow_backward:
            return self.step_chain(a, s)
        return self.step_dag(a, s)

    def step_dag(self, a, s=None):
        _s = s
        s = (self._state if s is None else s) + 0
        if a < self.ndim:
            s[a] += 1

        done = s.max() >= self.horizon - 1 or a == self.ndim
        if _s is None:
            self._state = s
            self._step += 1
        return self.obs(s), 0 if not done else self.func(self.s2x(s)), done, s

    def step_chain(self, a, s=None):
        _s = s
        s = (self._state if s is None else s) + 0
        sc = s + 0
        if a < self.ndim:
            s[a] = min(s[a]+1, self.horizon-1)
        if a >= self.ndim:
            s[a-self.ndim] = max(s[a-self.ndim]-1,0)

        reverse_a = ((a + self.ndim) % (2 * self.ndim)) if any(sc != s) else a

        if _s is None:
            self._state = s
            self._step += 1
        return self.obs(s), self.func(self.s2x(s)), s, reverse_a

    def true_density(self):
        if self._true_density is not None:  
            return self._true_density
        all_int_states = np.int32(list(itertools.product(*[list(range(self.horizon))]*self.ndim)))
        state_mask = np.array([len(self.parent_transitions(s, False)[0]) > 0 or sum(s) == 0
                               for s in all_int_states])
        all_xs = (np.float32(all_int_states) / (self.horizon-1) *
                  (self.xspace[-1] - self.xspace[0]) + self.xspace[0])
        traj_rewards = self.func(all_xs)[state_mask]
        self._true_density = (traj_rewards / traj_rewards.sum(),
                              list(map(tuple,all_int_states[state_mask])),
                              traj_rewards)
        return self._true_density

def make_mlp(l, act=nn.LeakyReLU(), tail=[]):
    """makes an MLP with no top layer activation"""
    return nn.Sequential(*(sum(
        [[nn.Linear(i, o)] + ([act] if n < len(l)-2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))

class ReplayBuffer:
    def __init__(self, env):
        self.buf = []
        self.env = env

    def load_data(self, data):
        self.buf = data

    def sample(self, sample_size):
        if not len(self.buf): 
            return []
        idxs = np.random.randint(0, len(self.buf), sample_size)
        result = [self.buf[i] for i in idxs]
        # return self.buf[idxs]
        return result

class PolicyBeta:
    def s2idx(self, s):
        s = s.reshape(self.ndim,self.horizon)
        idxs = s.argmax(-1)
        res,base = 0,1
        for i in range(len(idxs)):
            res += (idxs[i]) * base
            base *= self.horizon
            # print('idxs i',idxs[i],'res',res,'base',base)
        if isinstance(res, torch.Tensor):
            res=res.item()
        return res

    def __init__(self, args, replay,envs ):
        self.replay=replay
        self.envs = envs
        self.ndim=args.ndim
        self.horizon = args.horizon
        self.mp=set()
        for i in self.replay.buf:
            self.mp.add((self.s2idx(i[0]),self.s2idx(i[3])))
        print('test mp ok ',(0,1) in self.mp)
    
    def parent_transitions(self, s, used_stop_action):
        if used_stop_action:
            return [self.envs[0].obs(s)], [self.ndim]
        parents = []
        actions = []
        sidx=self.s2idx(self.envs[0].obs(s))
        for i in range(self.ndim):
            if s[i] > 0:
                sp = s + 0
                sp[i] -= 1
                spidx=self.s2idx(self.envs[0].obs(sp))
                if sp.max() == self.horizon-1: # can't have a terminal parent
                    continue
                if (spidx,sidx) not in self.mp:
                    continue
                parents += [self.envs[0].obs(sp)]
                actions += [i]
        return parents, actions
    
    def foward_mask(self, s):
        mask = []
        sidx=self.s2idx(self.envs[0].obs(s))
        for i in range(self.ndim):
            sp = s + 0
            if sp[i]<self.horizon-1:
                sp[i] += 1
                spidx=self.s2idx(self.envs[0].obs(sp))
                mask.append((sidx,spidx) in self.mp)
            else : mask.append(False)
        mask.append((sidx,sidx) in self.mp)
        return mask
            
        # for i in range(self)
    
class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant

class FlowNetAgent:
    def __init__(self, args, envs):
        self.model = make_mlp([args.horizon * args.ndim] +
                              [args.n_hid] * args.n_layers +
                              [args.ndim+1])
        self.model.to(args.dev)
        self.target = copy.deepcopy(self.model)
        self.envs = envs
        self.ndim = args.ndim
        self.tau = args.bootstrap_tau
        self.use_automatic_entropy_tuning= args.use_automatic_entropy_tuning
        self.alpha_multiplier = args.alpha_multiplier
        self.replay = ReplayBuffer(envs[0])
        loaded_data = torch.load(args.dataType)
        print(len(loaded_data));input()
        self.replay.load_data(loaded_data)
        self.policyBeta = PolicyBeta(args,self.replay,envs)

        if self.use_automatic_entropy_tuning>0:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=args.learning_rate,
            )
        else:
            self.log_alpha = Scalar(0.0)

    def _alpha_and_alpha_loss(self, log_pi: torch.Tensor):
        if self.use_automatic_entropy_tuning>0:
            alpha_loss = -(
                self.log_alpha() * (log_pi - (self.ndim+1) ).detach()
            ).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha , alpha_loss=0 , 0
        return alpha, alpha_loss

    def parameters(self):
        return self.model.parameters()
    
    def sample_many(self, mbsize, all_visited):
        batch = []
        s = tf([i.reset()[0] for i in self.envs])
        done = [False] * mbsize
        while not all(done):
            # Note to self: this is ugly, ugly code
            with torch.no_grad():
                acts = Categorical(logits=self.model(s)).sample()
            step = [i.step(a) for i,a in zip([e for d, e in zip(done, self.envs) if not d], acts)]
            # print('step',step,'mbsize',mbsize);input()
            p_a = [self.envs[0].parent_transitions(sp_state, a == self.ndim)
                   for a, (sp, r, done, sp_state) in zip(acts, step)]
            batch += [[tf(i) for i in (p, a, [r], [sp], [d])]
                      for (p, a), (sp, r, d, _) in zip(p_a, step)]
            c = count(0)
            m = {j:next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            s = tf([i[0] for i in step if not i[2]])
            for (_, r, d, sp) in step:
                if d:
                    # print('sp is',tuple(sp));input()
                    all_visited.append(tuple(sp))
        return batch
    
    def sample_replay(self, spsize):
        return self.replay.sample(spsize)

    def learn_from(self, batch):
        loginf = tf([1000])
        # parents, actions, r, sp, done = map(torch.cat, zip(*batch))
        _, a, r, sp, done = map(torch.cat, zip(*batch))
        p_a = [self.envs[0].parent_transitions(self.envs[0].obs2s(sp), a==self.ndim) for sp, a in zip(sp, a)]
        p_a = [ [tf(i) for i in (p,a) ] for (p,a) in p_a]
        p_a_beta = [self.policyBeta.parent_transitions(self.envs[0].obs2s(sp), a==self.ndim) for sp, a in zip(sp, a)]
        p_a_beta = [ [tf(i) for i in (p,a) ] for (p,a) in p_a_beta]
        parents,actions= map(torch.cat, zip(*p_a))
        parents_beta,actions_beta = map(torch.cat, zip(*p_a_beta))
        # print('parents_bata',len(parents_beta),'parents',len(parents));input()
        batch_idxs = tl(sum([[i]*len(parents) for i, (parents,_) in enumerate(p_a)], []))
        batch_idxs_beta = tl(sum([[i]*len(parents) for i, (parents,_) in enumerate(p_a_beta)], []))

        parents_Qsa = self.model(parents)[torch.arange(parents.shape[0]), actions.long()]
        parents_Qsa_beta = self.model(parents_beta)[torch.arange(parents_beta.shape[0]), actions_beta.long()]
        in_flow = torch.log(torch.zeros((sp.shape[0],))
                            .index_add_(0, batch_idxs, torch.exp(parents_Qsa)))
        in_flow_beta = torch.log(torch.zeros((sp.shape[0],))
                            .index_add_(0, batch_idxs_beta, torch.exp(parents_Qsa_beta)))
        
        if self.tau > 0:
            with torch.no_grad(): next_q = self.target(sp)
        else:
            next_q = self.model(sp)
            # print('next_q is',next_q);input()
        alpha, alpha_loss = self._alpha_and_alpha_loss(next_q)
        next_qd = next_q * (1-done).unsqueeze(1) + done.unsqueeze(1) * (-loginf)
        mask = [self.policyBeta.foward_mask(self.envs[0].obs2s(s)) for s in sp ]
        mask = torch.tensor(mask, dtype=torch.bool)
        next_qd_beta = next_q
        next_qd_beta[~mask] = -loginf
        next_qd_beta = next_qd_beta * (1-done).unsqueeze(1) + done.unsqueeze(1) * (-loginf)
        out_flow = torch.logsumexp(torch.cat([torch.log(r)[:, None], next_qd], 1), 1)
        out_flow_alpha = torch.logsumexp(next_qd, 1)
        out_flow_beta = torch.logsumexp(next_qd_beta, 1)
        # loss = (in_flow - out_flow).pow(2).mean()+ self.alpha*(in_flow-in_flow_beta).pow(2).mean() + self.alpha*(out_flow_alpha-out_floWw_beta).pow(2).mean()
        loss = (in_flow - out_flow).pow(2).mean()+ alpha*(in_flow-in_flow_beta).pow(2).mean() + alpha*(out_flow_alpha-out_flow_beta).pow(2).mean()

        if self.use_automatic_entropy_tuning>0:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        # loss = (in_flow - out_flow).pow(2).mean()

        with torch.no_grad():
            term_loss = ((in_flow - out_flow) * done).pow(2).sum() / (done.sum() + 1e-20)
            flow_loss = ((in_flow - out_flow) * (1-done)).pow(2).sum() / ((1-done).sum() + 1e-20)

        if self.tau > 0:
            for a,b in zip(self.model.parameters(), self.target.parameters()):
                b.data.mul_(1-self.tau).add_(self.tau*a)

        return loss, term_loss, flow_loss
    
def make_opt(params, args):
    params = list(params)
    if not len(params):
        return None
    if args.opt == 'adam':
        opt = torch.optim.Adam(params, args.learning_rate,
                               betas=(args.adam_beta1, args.adam_beta2))
    elif args.opt == 'msgd':
        opt = torch.optim.SGD(params, args.learning_rate, momentum=args.momentum)
    return opt

def compute_empirical_distribution_error(env, visited):
    if not len(visited):
        return 1, 100
    hist = defaultdict(int)
    for i in visited:
        hist[i] += 1
    td, end_states, true_r = env.true_density()
    true_density = tf(td)
    Z = sum([hist[i] for i in end_states])
    estimated_density = tf([hist[i] / Z for i in end_states])
    # print(estimated_density,true_density,end_states);input()
    k1 = abs(estimated_density - true_density).mean().item()
    # KL divergence
    kl = (true_density * torch.log((estimated_density+0.001) / (true_density+0.001))).mean().item()
    return k1, kl

def main(args):
    set_seed(2)
    args.dev = torch.device(args.device)
    set_device(args.dev)
    print('device', args.dev)
    f = {'default': None,
         'corners': func_corners,
    }[args.func]
    # 读取数据
    args.is_mcmc = args.method in ['mars', 'mcmc']

    env = GridEnv(args.horizon, args.ndim, func=f, allow_backward=args.is_mcmc)
    envs = [GridEnv(args.horizon, args.ndim, func=f, allow_backward=args.is_mcmc)
            for i in range(args.bufsize)]
    
    ndim = args.ndim
    if args.method == 'flownet':
        agent = FlowNetAgent(args, envs)

    opt = make_opt(agent.parameters(), args)
    exp_dir = f'{args.save_path}/{args.dataType}_{args.alpha_multiplier}'
    os.makedirs(exp_dir, exist_ok=True)

    # metrics
    all_losses = []
    all_visited = []
    empirical_distrib_losses = []
    modes_dict = {}
    last_idx = 0
    mode_dict = {k: False for k in np.ndindex(tuple([2,]*args.ndim))}

    ttsr = max(int(args.train_to_sample_ratio), 1)
    sttr = max(int(1/args.train_to_sample_ratio), 1) # sample to train ratio

    print('train begin')
    for i in tqdm(range(args.n_train_steps+1), disable=not args.progress):
        data = []
        for j in range(sttr):
            data += agent.sample_replay(args.mbsize*16)
            agent.sample_many(args.mbsize,all_visited)
        for j in range(ttsr):
            losses = agent.learn_from(data) # returns (opt loss, *metrics)
            if losses is not None:
                losses[0].backward()
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(agent.parameters(),
                                                   args.clip_grad_norm)
                opt.step()
                opt.zero_grad()
                all_losses.append([i.item() for i in losses])
        if i%100 == 0:
            smode_ls = [env.s2mode(s) for s in all_visited[last_idx:]]
            visited_mode_ls = [((smode+1)/2).astype(np.int32) for smode in smode_ls if smode.prod() != 0]
            for mode in visited_mode_ls:
                mode_dict[tuple(mode)] = True
            num_modes = len([None for k, v in mode_dict.items() if v is True])
            modes_dict[i] = num_modes
            print('epoch ',i,' loss', np.mean(losses[0].item()), 'nums of mode found is ',num_modes)
            last_idx = len(all_visited)

        if not i % 1000:
            # last_vis = []
            # for _ in range(625):
            #     agent.sample_many(args.mbsize, last_vis)
            empirical_distrib_losses.append(compute_empirical_distribution_error(env, all_visited[-50000:]))
            k1, kl = empirical_distrib_losses[-1]
            print('empirical L1 distance', k1, 'KL', kl)
            if len(all_losses):
                print(*[f'{np.mean([i[j] for i in all_losses[-100:]]):.5f}'
                        for j in range(len(all_losses[0]))])
            with open('losses.txt', 'a') as file:
                file.write(f'Epoch: {i}, empirical L1 distance: {k1}, cql_alpha is : {agent.log_alpha().exp() * agent.alpha_multiplier}\n')
            print('len all visited',len(all_visited))
            pickle.dump(
                all_visited,
                gzip.open(f'{exp_dir}/all_visited.pkl.gz', 'wb'))


    with open('losses.txt', 'a') as file:
        file.write(f'\n')

    root = os.path.split(args.save_path)[0]
    os.makedirs(root, exist_ok=True)
    # pickle.dump(
    #     {'losses': np.float32(all_losses),
    #      #'model': agent.model.to('cpu') if agent.model else None,
    #      'params': [i.data.to('cpu').numpy() for i in agent.parameters()],
    #      'visited': np.int8(all_visited),
    #      'emp_dist_loss': empirical_distrib_losses,
    #      'true_d': env.true_density()[0],
    #      'args':args},
    #     gzip.open(args.save_path, 'wb'))
    pickle.dump(
        all_visited,
        gzip.open(f'{args.save_path}/all_visited.pkl.gz', 'wb'))

if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args)
