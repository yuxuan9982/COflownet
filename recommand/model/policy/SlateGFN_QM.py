import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

from model.general import BaseModel
from model.components import DNN
from model.policy.BaseOnlinePolicy import BaseOnlinePolicy
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
import math

def make_mlp(l, act=nn.ReLU(), tail=[], tailNorm= True):
    """makes an MLP with no top layer activation"""
    if tailNorm==False:
        net = nn.Sequential(*(sum(
            [[nn.Linear(i, o)] + ([act] if n < len(l)-2 else []) + ([nn.Dropout(0.1)] if n < len(l)-2 else []) + ([nn.LayerNorm([o])] if n < len(l)-2 else [])
            for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))
    else:
        net = nn.Sequential(*(sum(
            [[nn.Linear(i, o)] + ([act] if n < len(l)-1 else []) + ([nn.Dropout(0.1)] if n < len(l)-1 else []) + ([nn.LayerNorm([o])] if n < len(l)-1 else [])
            for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))
    return net

class IQN(nn.Module):
    def __init__(self, in_dim, hidden_dim , quantile_dim=256 , out_dim = 3043):
        super().__init__()
        self.feature = make_mlp([in_dim] + [hidden_dim] )
        self.quantile_embed_dim = quantile_dim
        self.phi = make_mlp([self.quantile_embed_dim,] + [hidden_dim]*2)
        self.register_buffer("feature_id", torch.arange(1, 1+self.quantile_embed_dim))
        self.lastDrop = nn.Dropout(0.1)
        self.lastNorm = nn.LayerNorm(hidden_dim)
        self.last = nn.Linear(hidden_dim,out_dim)
        # self.last = make_mlp([hidden_dim]*2+ [out_dim],tailNorm=False)
        
    def forward(self, state, quantiles):
        batch_size, n_quantiles = quantiles.shape
        assert batch_size == state.shape[0]
        feature_id = repeat(self.feature_id, "d -> b n d", b=batch_size, n=n_quantiles)
        quantiles_rep = repeat(quantiles, "b n -> b n d", d=self.quantile_embed_dim)
        cos = torch.cos(math.pi * feature_id * quantiles_rep) # (bs, n_quantiles, d)
        x = self.feature(state).unsqueeze(1) * F.relu(self.phi(cos))
        x = self.lastDrop(x)
        x = self.lastNorm(x)
        logflow_vals = self.last(x) # (bs, n_quantiles, ndim+1)
        return logflow_vals
        

class SlateGFN_QM(BaseOnlinePolicy):
    '''
    GFlowNet with Detailed Balance for listwise recommendation
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - gfn_forward_hidden_dims
        - gfn_flow_hidden_dims
        - gfn_forward_offset
        - gfn_reward_smooth
        - gfn_Z
        - from BaseOnlinePolicy:
            - from BackboneUserEncoder:
                - user_latent_dim
                - item_latent_dim
                - transformer_enc_dim
                - transformer_n_head
                - transformer_d_forward
                - transformer_n_layer
                - state_hidden_dims
                - dropout_rate
                - from BaseModel:
                    - model_path
                    - loss
                    - l2_coef
        '''
        parser = BaseOnlinePolicy.parse_model_args(parser) 
        parser.add_argument('--gfn_forward_hidden_dims', type=int, nargs="+", default=[128], 
                            help='hidden dimensions of state_slate encoding layers')
        parser.add_argument('--gfn_flow_hidden_dims', type=int, nargs="+", default=[128], 
                            help='hidden dimensions of flow estimator')
        parser.add_argument('--gfn_forward_offset', type=float, default=1.0, 
                            help='smooth offset of forward logp of TB loss')
        parser.add_argument('--gfn_reward_smooth', type=float, default=1.0, 
                            help='reward smooth offset in the backward part of TB loss')
        parser.add_argument('--gfn_Z', type=float, default=0., 
                            help='average reward offset')
        
        return parser
        
    def __init__(self, args, reader_stats, device):
        # BaseModel initialization: 
        # - reader_stats, model_path, loss_type, l2_coef, no_reg, device, slate_size
        # - _define_params(args)
        self.gfn_forward_hidden_dims = args.gfn_forward_hidden_dims
        self.gfn_flow_hidden_dims = args.gfn_flow_hidden_dims
        self.gfn_forward_offset = args.gfn_forward_offset
        self.gfn_reward_smooth = args.gfn_reward_smooth
        self.gfn_Z = args.gfn_Z
        super().__init__(args, reader_stats, device)
        self.display_name = "GFN_QM"
        self.alpha=0.1
        self.N = 8
        self.alpha_decrease= self.alpha/2500
        
    def to(self, device):
        new_self = super(SlateGFN_QM, self).to(device)
        return new_self

    def _define_params(self, args):
        # userEncoder, enc_dim, state_dim, bce_loss
        super()._define_params(args)
        # p_forward
        # self.pForwardEncoder = make_mlp([self.state_dim + self.enc_dim * self.slate_size]+args.gfn_forward_hidden_dims+[3043])
        self.model = IQN(self.state_dim + self.enc_dim * self.slate_size, 128, out_dim=3043)

    
    def generate_action(self, user_state, feed_dict):
        candidates = feed_dict['candidates']
        slate_size = feed_dict['action_dim']
        parent_slate = feed_dict['action'] # (B, K)
        do_explore = feed_dict['do_explore']
        do_explore = True
        is_train = feed_dict['is_train']
        epsilon = feed_dict['epsilon']
        future = None
        if 'future' in feed_dict:
            future = feed_dict['future']
        '''
        @input:
        - user_state: (B, state_dim) 
        - feed_dict: same as BaseOnlinePolicy.get_forward@feed_dict
        @output:
        - out_dict: {'prob': (B, K), 
                     'logF': (B,),
                     'action': (B, K), 
                     'reg': scalar}
        '''
        B = user_state.shape[0]
        # batch-wise candidates has shape (B,L), non-batch-wise candidates has shape (1,L)
        batch_wise = True
        if candidates['item_id'].shape[0] == 1:
            batch_wise = False
        # during training, candidates is always the full item set and has shape (1,L) where L=N
        if is_train:
            assert not batch_wise
        # epsilon probability for uniform sampling under exploration
        do_uniform = np.random.random() < epsilon
            
        # (1,L,enc_dim)
        candidate_item_enc, reg = self.userEncoder.get_item_encoding(candidates['item_id'], 
                                                       {k[5:]: v for k,v in candidates.items() if k != 'item_id'}, 
                                                                     B if batch_wise else 1)
        
        # forward probabilities P(a_t|s_{t-1}) of the action, size (B, K)
        current_P = torch.zeros(B, slate_size).to(self.device)
        # (B, K)
        current_action = torch.zeros(B, slate_size).to(torch.long).to(self.device)
        # (B, K, enc_dim)
        current_list_emb = torch.zeros(B, slate_size, self.enc_dim).to(self.device)
        # (B, K+1)
        # current_flow = torch.zeros(B, slate_size + 1).to(self.device)
        # (B, K)
        current_in_flow = torch.zeros(B, self.N , slate_size).to(self.device)
        # (B, K)
        current_out_flow = torch.zeros(B,self.N, slate_size).to(self.device)
        #（B, K）
        support_out_flow = torch.zeros(B,self.N, slate_size).to(self.device)
        # support_future_flow = torch.zeros(B, slate_size).to(self.device)
        # regressive action generation
        in_quantiles = torch.rand(B, self.N, slate_size).to(self.device)
        out_quantiles = torch.rand(B, self.N, slate_size).to(self.device)
        for i in range(slate_size):
            # (B, state_dim + slate_size * enc_dim)
            current_state = torch.cat((user_state.view(B, self.state_dim), current_list_emb.view(B, -1)), dim = 1)
            # (B, enc_dim)
            # selection_weight = self.pForwardEncoder(current_state)
            # (B, L)
            # score = torch.sum(selection_weight.view(B,1,self.enc_dim) * candidate_item_enc, dim = -1) #/ self.enc_dim
            # (B, L)
            model_out = self.model(current_state, in_quantiles[:,:,i].view(B,-1))#doubt right?
            next_q = self.model(current_state, out_quantiles[:,:,i].view(B,-1))
            score = model_out
            prob = torch.softmax(score.logsumexp(dim=1), dim = 1)
            # (B,)
            zero_column = torch.full((next_q.shape[0], next_q.shape[1], 1), -1000.0).to(self.device)
            next_q = torch.cat((next_q,zero_column),dim=-1)
            # print(score.shape)
            # if is_train or torch.is_tensor(parent_slate):
            if is_train:
                # during training, output the target action probability without sampling
                action_at_i = parent_slate[:,i] # (B,)
                current_list_emb[:,i,:] = candidate_item_enc.view(-1,self.enc_dim)[action_at_i]
                action_repeat_at_i = repeat(action_at_i.view(-1,1),'a b -> a c b', c=score.shape[1])
                current_in_flow[:,:,i] = torch.gather(score,2,action_repeat_at_i).view(B,-1)#[128,8,1]->[128,8]
                # print(current_in_flow[:,:,i].shape);assert 0
                current_out_flow[:,:,i] = torch.logsumexp(next_q,2)
                current_action[:,i] = action_at_i
                # supported Probabilities
                action_at_and_after_i = parent_slate[:,i:]
                action_with_future = torch.cat((action_at_and_after_i,future),1)
                action_repeat_future_i = repeat(action_with_future,'a b -> a c b', c=score.shape[1])
                support_flow = torch.gather(next_q,2,action_repeat_future_i)
                # sum_flow = torch.logsumexp(support_P,1)
                support_out_flow[:,:,i]= torch.logsumexp(support_flow,2)
                # print('slate ok ',i)
            else:
                if i > 0:
                    # remove items already selected
                    prob.scatter_(1,current_action[:,:i],0)
             
                if do_explore:
                    # exploration: categorical sampling or uniform sampling
                    if do_uniform:
                        indices = Categorical(torch.ones_like(prob)).sample()
                    else:
                        indices = Categorical(prob).sample()
                else: 
                    # greedy: topk selection
                    _, indices = torch.topk(prob, k = 1, dim = 1)
                indices = indices.view(-1).detach()
                # update current slate action
                current_action[:,i] = indices
                # update slate action probability
                current_P[:,i] = torch.gather(prob,1,indices.view(-1,1)).view(-1)

                if batch_wise:
                    for j in range(B):
                        current_list_emb[j,i,:] = candidate_item_enc[j,indices[j]]
                else:
                    current_list_emb[:,i,:] = candidate_item_enc.view(-1,self.enc_dim)[indices]
        if is_train:
            current_state = torch.cat((user_state.view(B, self.state_dim), current_list_emb.view(B, -1)), dim = 1)
            reg = 0
        else:
            reg = 0

        # print(support_sum_p,support_sum_p.shape);input()
        # print(current_in_flow,current_out_flow,support_out_flow,reg)
        out_dict = {'prob': current_P, 
                    'action': current_action, 
                    'log_in_F': current_in_flow, 
                    'log_out_F': current_out_flow, 
                    'support_out_F':support_out_flow,
                    'in_quantiles':in_quantiles,
                    'out_quantiles':out_quantiles,
                    # 'support_future_F':support_future_flow,
                    'reg': reg
                    }
        return out_dict
    
    def get_loss(self, feed_dict, out_dict):
        '''
        Detailed balance loss (Note: log(P(s[t-1]|s[t])) = 0 if tree graph): 
        * non-terminal: ( log(flow(s[t-1])) + log(P(s[t]|s[t-1])) - log(flow(s[t])) )^2
        * terminal: ( log(flow(s[t])) - log(reward(s[t])) )^2
        
        @input:
        - feed_dict: same as BaseOnlinePolicy.get_forward@input-feed_dict
        - out_dict: {
            'state': (B,state_dim), 
            'prob': (B,K),
            'logF': (B,)
            'action': (B,K),
            'reg': scalar, 
            'immediate_response': (B,K*n_feedback),
            'reward': (B,)}
        @output
        - loss
        '''
        # (B, K)
        parent_flow = out_dict['log_in_F']
        # (B, K)
        current_flow = out_dict['log_out_F']
        # (B, K)
        support_flow = out_dict['support_out_F']
        # (B, N)
        in_quantiles = out_dict['in_quantiles']
        # (B, N)
        out_quantiles = out_dict['out_quantiles']
        # support_future_flow = out_dict['support_future_F']
        # (B, K)
        forward_part = parent_flow[:,:,:]
        # (B, K)
        # backward_part = current_flow[:,:,1:]
        backward_part = torch.cat((current_flow[:,:,1:],repeat(torch.log(out_dict['reward'] + self.gfn_reward_smooth + 1e-6).view(-1,1),'b 1-> b nq 1',nq=self.N)), 2 )
        # scalar
        diff = repeat(backward_part,'b nout slate -> b 1 nout slate') - repeat(forward_part,'b nin slate -> b nin 1 slate')
        abs_weight = torch.abs(repeat(in_quantiles, "b n_in slate -> b n_in n_out slate", n_out=self.N)  - diff.le(0).float())
        # (B, )
        # diff_losses = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none")
        diff_losses = diff.pow(2)
        QM_loss = (abs_weight * diff_losses).sum(dim=1).mean(dim=2) # sum over qunatile_in, mean over quantile_out
        # CO_loss = torch.mean((current_flow[:,:].detach()-support_flow[:,:]))
        out_diff = current_flow.detach() - support_flow
        # out_diff_losses = torch.nn.functional.smooth_l1_loss(out_diff, torch.zeros_like(out_diff), reduction="none")
        out_abs_weight = torch.abs(out_quantiles - out_diff.le(0).float())
        CO_loss = (out_abs_weight*out_diff).sum(dim=1) 
        # scalar
        loss = torch.mean(QM_loss) + 1.0*torch.mean(CO_loss)
        # loss = torch.mean(QM_loss) + 1.0*torch.mean(CO_loss)
        
        return {'loss': loss, 'QM_loss': torch.mean(QM_loss), 'terminal_loss': loss, 'CO_loss': torch.mean(CO_loss), 
                'forward_part': torch.mean(forward_part), 'backward_part': torch.mean(backward_part), 
                'prob': torch.mean(out_dict['prob']),'support_flow':torch.mean(support_flow),'current_flow':torch.mean(current_flow)}

    def get_loss_observation(self):
        return ['loss', 'QM_loss', 'terminal_loss', 'CO_loss', 'forward_part', 'backward_part', 'prob','support_flow','current_flow']
        
        
        