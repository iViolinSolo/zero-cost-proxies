'''
Author: ViolinSolo
Date: 2023-04-07 18:11:15
LastEditTime: 2023-04-07 18:13:07
LastEditors: ViolinSolo
Description: grasp from pruners/measures/grasp.py
FilePath: /zero-cost-proxies/alethiometer/zero_cost_metrics/grasp.py
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from . import metric
from ..utils import get_layer_metric_array


@metric('grasp', bn=True, mode='param')
def compute_grasp_per_weight(net, inputs, targets, mode, loss_fn, T=1, num_iters=1, split_data=1):

    # get all applicable weights
    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weights.append(layer.weight)
            layer.weight.requires_grad_(True) # TODO isn't this already true?

    # NOTE original code had some input/target splitting into 2
    # I am guessing this was because of GPU mem limit
    net.zero_grad()
    N = inputs.shape[0]
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data

        #forward/grad pass #1
        grad_w = None
        for _ in range(num_iters):
            #TODO get new data, otherwise num_iters is useless!
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
            outputs = outputs/T
            loss = loss_fn(outputs, targets[st:en])
            grad_w_p = autograd.grad(loss, weights, allow_unused=True)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]

    
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data

        # forward/grad pass #2
        outputs = net.forward(inputs[st:en])
        if type(outputs) is tuple:
            _, outputs = outputs
        outputs = outputs/T
        loss = loss_fn(outputs, targets[st:en])
        grad_f = autograd.grad(loss, weights, create_graph=True, allow_unused=True)
        
        # accumulate gradients computed in previous step and call backwards
        z, count = 0,0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if grad_w[count] is not None:
                    z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
        z.backward()

    # compute final sensitivity metric and put in grads
    def grasp(layer):
        if layer.weight.grad is not None:
            return -layer.weight.data * layer.weight.grad   # -theta_q Hg
            #NOTE in the grasp code they take the *bottom* (1-p)% of values
            #but we take the *top* (1-p)%, therefore we remove the -ve sign
            #EDIT accuracy seems to be negatively correlated with this metric, so we add -ve sign here!
        else:
            return torch.zeros_like(layer.weight)
    
    grads = get_layer_metric_array(net, grasp, mode)

    return grads