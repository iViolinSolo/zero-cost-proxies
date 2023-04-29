'''
Author: ViolinSolo
Date: 2023-04-29 17:33:33
LastEditTime: 2023-04-29 18:18:01
LastEditors: ViolinSolo
Description: t_cet metric
FilePath: /zero-cost-proxies/alethiometer/zero_cost_metrics/t_cet.py
'''

import numpy as np
import torch

from . import metric
from .naswot import compute_naswot
from .synflow import compute_synflow_per_weight
from .snip import compute_snip_per_weight


@metric('tcet_syn_none', bn=True, mode='none', mt='synflow')
@metric('tcet_snip_none', bn=True, mode='none', mt='snip')
@metric('tcet_syn_log', bn=True, mode='log', mt='synflow')
@metric('tcet_snip_log', bn=True, mode='log', mt='snip')
@metric('tcet_syn_log1p', bn=True, mode='log1p', mt='synflow')
@metric('tcet_snip_log1p', bn=True, mode='log1p', mt='snip')
@metric('tcet_syn_norm', bn=True, mode='norm', mt='synflow')
@metric('tcet_snip_norm', bn=True, mode='norm', mt='snip')
@metric('tcet', bn=True, mode='none', mt='synflow')
def compute_tcet_score(net, inputs, targets, loss_fn=None, split_data=1,
                      mode='none', mt='synflow'):
    
    if mode not in ['none', 'log', 'log1p', 'norm']:
        raise ValueError('mode {} not supported for t_cet'.format(mode))
    
    if mt not in ['synflow', 'snip']:
        raise ValueError('mt {} not supported for t_cet'.format(mt))
    
    # compute layerwise nwot scores
    lnwot_scores = compute_naswot(net, inputs, targets, loss_fn, split_data, layerwise=True, return_Kmats=False)

    # compute layerwise mt scores
    mt_scores = None
    if mt == 'synflow':
        synflow_scores = compute_synflow_per_weight(net, inputs, targets, loss_fn, split_data, mode='param')
        mt_scores = synflow_scores
    elif mt == 'snip':
        snip_scores = compute_snip_per_weight(net, inputs, targets, loss_fn, split_data, mode='param')
        mt_scores = snip_scores
    
    # compute snr mt scores
    snr_mt_scores = []
    for mt_score in mt_scores:
        s = mt_score.detach().view(-1)

        sigma = torch.std(s)
        sigma = torch.tensor(0.) if torch.isnan(sigma) else sigma  ###TODO: Notice, torch.std(torch.tensor([2.2]))==tensor(nan), np.std([2.2])==0.
        
        if sigma == 0 or torch.isnan(sigma):
            s = torch.sum(s)
        else:
            s = torch.sum(s)/sigma
            
        s_val = s.cpu().item()
        snr_mt_scores.append(s_val)


    _mt_std = np.std(snr_mt_scores) if mode == 'norm' else 0.
    # compute layerwise t_cet scores
    tcet_scores = []
    for lnwot_score, s_val in zip(lnwot_scores, snr_mt_scores):
        if mode == 'none':
            s_val = s_val
        elif mode == 'log':
            s_val = np.log(s_val) if s_val>0 else 0.
        elif mode == 'log1p':
            s_val = np.log1p(s_val) if s_val>=0 else 0.
        elif mode == 'norm':
            s_val = (s_val / _mt_std) if _mt_std>0 else s_val

        l_val = lnwot_score.cpu().item()
        tcet_score = l_val * s_val
        tcet_scores.append(tcet_score)
    
    return tcet_scores

