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


parser = argparse.ArgumentParser()

parser.add_argument("--save_path", default='results/flow_insp_0.pkl.gz', type=str)
parser.add_argument("--device", default='cpu', type=str)
parser.add_argument("--progress", action='store_true')

#
parser.add_argument("--method", default='flownet', type=str)
parser.add_argument("--learning_rate", default=1e-4, help="Learning rate", type=float)
parser.add_argument("--opt", default='adam', type=str)
parser.add_argument("--adam_beta1", default=0.9, type=float)
parser.add_argument("--adam_beta2", default=0.999, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--mbsize", default=16, help="Minibatch size", type=int)
parser.add_argument("--train_to_sample_ratio", default=1, type=float)
parser.add_argument("--n_hid", default=256, type=int)
parser.add_argument("--n_layers", default=2, type=int)
parser.add_argument("--n_train_steps", default=20000, type=int)
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
parser.add_argument("--random", action='store_true')




_dev = [torch.device('cpu')]
tf = lambda x: torch.FloatTensor(x).to(_dev[0])
tl = lambda x: torch.LongTensor(x).to(_dev[0])

def set_device(dev):
    _dev[0] = dev


def func_corners(x):
    ax = abs(x)
    return (ax > 0.5).prod(-1) * 0.5 + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2 + 1e-5

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

    def reward(self, s):#其他地方拿过来的reward!!!!
        self.size=16
        grid = s.view(len(s), self.size, self.size)
        coord = (grid == 1).nonzero()[:, 1:].view(len(s), 2)
        R0, R1, R2 = 1e-2, 0.5, 2
        norm = torch.abs(coord / (self.size-1) - 0.5)
        R1_term = torch.prod(0.25 < norm, dim=1)
        R2_term = torch.prod((0.3 < norm) & (norm < 0.4), dim=1)
        return (R0 + R1*R1_term + R2*R2_term)

    def obs(self, s=None):
        s = np.int32(self._state if s is None else s)
        z = np.zeros((self.horizon * self.ndim), dtype=np.float32)
        z[np.arange(len(s)) * self.horizon + s] = 1
        return z

    def s2x(self, s):
        return (self.obs(s).reshape((self.ndim, self.horizon)) * self.xspace[None, :]).sum(1)

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
        return self.step_dag(a)

    def step_dag(self, a, s=None):
        _s = s
        s = (self._state if s is None else s) + 0
        s_unchanged = s + 0
        if a < self.ndim:
            s[a] += 1

        done = s.max() >= self.horizon - 1 or a == self.ndim
        if _s is None:
            self._state = s
            self._step += 1
        return self.obs(s), 0 if not done else self.func(self.s2x(s)), done, s, self.obs(s_unchanged)  

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
    def __init__(self, args, env):
        self.buf = []
        self.strat = args.replay_strategy
        self.sample_size = args.replay_sample_size
        self.bufsize = args.replay_buf_size
        self.env = env

    def add(self, x, r_x):
        if self.strat == 'top_k':
            if len(self.buf) < self.bufsize or r_x > self.buf[0][0]:
                self.buf = sorted(self.buf + [(r_x, x)])[-self.bufsize:]

    def sample(self):
        if not len(self.buf):
            return []
        idxs = np.random.randint(0, len(self.buf), self.sample_size)
        return sum([self.generate_backward(*self.buf[i]) for i in idxs], [])

    def generate_backward(self, r, s0):
        s = np.int8(s0)
        os0 = self.env.obs(s)
        # If s0 is a forced-terminal state, the the action that leads
        # to it is s0.argmax() which .parents finds, but if it isn't,
        # we must indicate that the agent ended the trajectory with
        # the stop action
        used_stop_action = s.max() < self.env.horizon - 1
        done = True
        # Now we work backward from that last transition
        traj = []
        while s.sum() > 0:
            parents, actions = self.env.parent_transitions(s, used_stop_action)
            # add the transition
            traj.append([tf(i) for i in (parents, actions, [r], [self.env.obs(s)], [done])])
            # Then randomly choose a parent state
            if not used_stop_action:
                i = np.random.randint(0, len(parents))
                a = actions[i]
                s[a] -= 1
            # Values for intermediary trajectory states:
            used_stop_action = False
            done = False
            r = 0
        return traj

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
        self.random = args.random
        self.replay = ReplayBuffer(args, envs[0])

    def parameters(self):
        return self.model.parameters()

    def sample_many(self, mbsize, all_visited, generate_data=False, force = False):
        batch = []
        batch += self.replay.sample()
        s = tf([i.reset()[0] for i in self.envs])
        done = [False] * mbsize
        offline_data = []
        while not all(done):
            # Note to self: this is ugly, ugly code
            with torch.no_grad():
                if self.random or False:
                    logits = torch.randn(16, 5)
                    acts = Categorical(logits=logits).sample()
                else:
                    acts = Categorical(logits=self.model(s)).sample()

            step = [i.step(a) for i,a in zip([e for d, e in zip(done, self.envs) if not d], acts)]
            # print('step',step,'mbsize',mbsize);input()
            p_a = [self.envs[0].parent_transitions(sp_state, a == self.ndim)
                   for a, (sp, r, done, sp_state,_) in zip(acts, step)]
            batch += [[tf(i) for i in (p, a, [r], [sp], [d])]
                      for (p, a), (sp, r, d, _, _) in zip(p_a, step)]
            # print('batch is',batch);input()
            offline_data+=[[tf(i) for i in ([s], [a], [r], [sp], [d])]
                      for a, (sp, r, d, _, s) in zip(acts, step)]
            c = count(0)
            m = {j:next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            s = tf([i[0] for i in step if not i[2]])
            for (_, r, d, sp, _) in step:
                if d:
                    # print('sp is',tuple(sp));input()
                    all_visited.append(tuple(sp))
                    self.replay.add(tuple(sp), r)
        if generate_data:
            return offline_data
        return batch


    def learn_from(self, it, batch):
        loginf = tf([1000])
        batch_idxs = tl(sum([[i]*len(parents) for i, (parents,_,_,_,_) in enumerate(batch)], []))
        parents, actions, r, sp, done = map(torch.cat, zip(*batch))
        parents_Qsa = self.model(parents)[torch.arange(parents.shape[0]), actions.long()]
        in_flow = torch.log(torch.zeros((sp.shape[0],))
                            .index_add_(0, batch_idxs, torch.exp(parents_Qsa)))
        if self.tau > 0:
            with torch.no_grad(): next_q = self.target(sp)
        else:
            next_q = self.model(sp)
        next_qd = next_q * (1-done).unsqueeze(1) + done.unsqueeze(1) * (-loginf)
        # print(torch.cat([torch.log(r)[:, None], next_qd], 1));input()
        out_flow = torch.logsumexp(torch.cat([torch.log(r)[:, None], next_qd], 1), 1)
        loss = (in_flow - out_flow).pow(2).mean()

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
    k1 = abs(estimated_density - true_density).mean().item()
    # KL divergence
    kl = (true_density * torch.log(estimated_density / true_density)).sum().item()
    return k1, kl

def main(args):
    args.dev = torch.device(args.device)
    set_device(args.dev)
    print('device', args.dev)
    f = {'default': None,
         'corners': func_corners,
    }[args.func]

    args.is_mcmc = args.method in ['mars', 'mcmc']

    env = GridEnv(args.horizon, args.ndim, func=f, allow_backward=args.is_mcmc)
    envs = [GridEnv(args.horizon, args.ndim, func=f, allow_backward=args.is_mcmc)
            for i in range(args.bufsize)]
    ndim = args.ndim

    if args.method == 'flownet':
        agent = FlowNetAgent(args, envs)

    opt = make_opt(agent.parameters(), args)

    # metrics
    all_losses = []
    all_visited = []
    empirical_distrib_losses = []

    ttsr = max(int(args.train_to_sample_ratio), 1)
    sttr = max(int(1/args.train_to_sample_ratio), 1) # sample to train ratio

    if args.method == 'ppo':
        ttsr = args.ppo_num_epochs
        sttr = args.ppo_epoch_size

    print('train begin')
    for i in tqdm(range(args.n_train_steps+1), disable=not args.progress):
        data = []
        for j in range(sttr):
            data += agent.sample_many(args.mbsize, all_visited)
        for j in range(ttsr):
            losses = agent.learn_from(i * ttsr + j, data) # returns (opt loss, *metrics)
            if losses is not None:
                losses[0].backward()
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(agent.parameters(),
                                                   args.clip_grad_norm)
                opt.step()
                opt.zero_grad()
                all_losses.append([i.item() for i in losses])

        if i%100 == 0:
            print('epoch ',i,' loss', np.mean(losses[0].item()))
        if not i % 100:
            last_vis = []
            empirical_distrib_losses.append(
                compute_empirical_distribution_error(env, all_visited[-args.num_empirical_loss:]))
            if args.progress:
                k1, kl = empirical_distrib_losses[-1]
                print('empirical L1 distance', k1, 'KL', kl)
                if len(all_losses):
                    print(*[f'{np.mean([i[j] for i in all_losses[-100:]]):.5f}'
                            for j in range(len(all_losses[0]))])

    root = os.path.split(args.save_path)[0]
    os.makedirs(root, exist_ok=True)
    pickle.dump(
        {'losses': np.float32(all_losses),
         #'model': agent.model.to('cpu') if agent.model else None,
         'params': [i.data.to('cpu').numpy() for i in agent.parameters()],
         'visited': np.int8(all_visited),
         'emp_dist_loss': empirical_distrib_losses,
         'true_d': env.true_density()[0],
         'args':args},
        gzip.open(args.save_path, 'wb'))
    last_vis = []
    # last_sum = [[0 for i in range(16)] for j in range(16)]
    batch_lst = []
    while len(batch_lst)<0:
        batch = agent.sample_many(args.mbsize, last_vis,True)
        batch_lst.extend(batch)
    while len(batch_lst)<20000:
        batch = agent.sample_many(args.mbsize, last_vis,True,True)
        batch_lst.extend(batch)
    # print('batch_lst',batch_lst[0])
    torch.save(batch_lst, 'random.pt')
    print('save successful')
    

if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args)
