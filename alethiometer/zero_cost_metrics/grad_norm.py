'''
Author: ViolinSolo
Date: 2023-04-07 14:16:41
LastEditTime: 2023-04-07 18:10:09
LastEditors: ViolinSolo
Description: snip from pruners/measures/snip.py
FilePath: /zero-cost-proxies/alethiometer/zero_cost_metrics/grad_norm.py
'''

import torch

from . import metric
from ..utils import get_layer_metric_array

@metric('grad_norm', bn=True)
def get_grad_norm_arr(net, inputs, targets, loss_fn, split_data=1, skip_grad=False):
    net.zero_grad()
    N = inputs.shape[0]
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data

        outputs = net.forward(inputs[st:en])
        if type(outputs) is tuple:
            # outputs, _ = outputs #original code
            # original code don't came into bug, is original input feat into nn.cross_entroy
            # which means, it mistakenly identify n_class from original 10[cifar10], to 64.
            # feat, logits = outputs
            # feat.shape = [64, 64]
            # logits.shape = [64, 10]
            # nn.cross_entropy(inputs, targets) takes: input [N, C] and targets [N, ] of class_index or class_prob
            _, outputs = outputs #TODO: need logits just like synflow and snip, so fix here temporarily
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()

        grad_norm_arr = get_layer_metric_array(net, lambda l: l.weight.grad.norm() if l.weight.grad is not None else torch.zeros_like(l.weight), mode='param')
        
    return grad_norm_arr