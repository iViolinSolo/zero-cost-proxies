'''
Author: ViolinSolo
Date: 2023-04-07 14:26:18
LastEditTime: 2023-04-07 16:00:56
LastEditors: ViolinSolo
Description: utils selected from pruners/p_utils.py
FilePath: /zero-cost-proxies/alethiometer/utils.py
'''

# import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F


def get_layer_metric_array(net, metric, mode): 
    metric_array = []

    for layer in net.modules():
        if mode=='channel' and hasattr(layer,'dont_ch_prune'):
            continue
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv1d):
            metric_array.append(metric(layer))
    
    return metric_array


def get_some_data(train_dataloader, num_batches=1, device=None):
    traindata = []
    dataloader_iter = iter(train_dataloader)
    for _ in range(num_batches):
        traindata.append(next(dataloader_iter))
    inputs  = torch.cat([a for a,_ in traindata])
    targets = torch.cat([b for _,b in traindata])
    inputs = inputs.to(device)
    targets = targets.to(device)
    return inputs, targets