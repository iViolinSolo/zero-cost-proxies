'''
Author: ViolinSolo
Date: 2023-04-06 18:35:04
LastEditTime: 2023-04-11 13:49:23
LastEditors: ViolinSolo
Description: entry program
FilePath: /zero-cost-proxies/alethiometer/zc_proxy.py
'''
import types
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import numpy as np


from . import zero_cost_metrics as M
from .utils import get_some_data


ZC_METRIC_SUPPORTED = [
    '',

]

def no_op(self, x):
    return x

def copynet(self, bn):
    net = copy.deepcopy(self)
    if bn==False:
        for l in net.modules():
            if isinstance(l,nn.BatchNorm2d) or isinstance(l,nn.BatchNorm1d) :
                l.forward = types.MethodType(no_op, l)
    return net


def calc_vals(net_orig, trainloader, device, metric_names:list, loss_fn=F.cross_entropy):

    if not hasattr(net_orig, 'get_prunable_copy'):
        net_orig.get_prunable_copy = types.MethodType(copynet, net_orig)

    #move to cpu to free up mem
    torch.cuda.empty_cache()
    net_orig = net_orig.cpu() 
    torch.cuda.empty_cache()

    #given 1 minibatch of data
    inputs, targets = get_some_data(trainloader, num_batches=1, device=device)

    done, ds = False, 1
    metric_vals = {}

    while not done:
        try:
            for mt_name in metric_names:
                if mt_name not in metric_vals:
                    val = M.calc_metric(mt_name, net_orig, device, inputs, targets, loss_fn=loss_fn, split_data=ds)
                    metric_vals[mt_name] = val

            done = True
        except RuntimeError as e:
            if 'out of memory' in str(e):
                done=False
                if ds == inputs.shape[0]//2:
                    raise ValueError(f'Can\'t split data anymore, but still unable to run. Something is wrong') 
                ds += 1
                while inputs.shape[0] % ds != 0:
                    ds += 1
                torch.cuda.empty_cache()
                print(f'Caught CUDA OOM, retrying with data split into {ds} parts')
            else:
                raise e

    net_orig = net_orig.to(device).train()
    return metric_vals


def calc_zc_metrics(metrics: list, model: nn.Module, train_queue: D.DataLoader, device: torch.device, loss_fn=F.cross_entropy, aggregate=True):
    """
    Purpose: metrics calculation entry.
    @param: train_queue: train dataset dataloader.
    @param: aggregate: whether return original layerwise value, when true, return processed aggregated value.

    @return: dict of values, key is metric name of @params metrics, value is the calculated zcmetric result, 
    """
    mt_vals = calc_vals(net_orig=model, trainloader=train_queue, device=device, metric_names=metrics, loss_fn=loss_fn)

    def sum_arr(arr):
        sum = 0.
        for i in range(len(arr)):
            sum += torch.sum(arr[i])
        return sum.item()
    
    results = {}
    for k, v in mt_vals.items():
        results[k] = v if not aggregate else sum_arr(v)

    return results
# end def