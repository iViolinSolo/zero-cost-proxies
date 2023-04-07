'''
Author: ViolinSolo
Date: 2023-04-07 14:16:23
LastEditTime: 2023-04-07 18:06:05
LastEditors: ViolinSolo
Description: synflow from pruners/measures/synflow.py
FilePath: /zero-cost-proxies/alethiometer/zero_cost_metrics/synflow.py
'''

import torch
from . import metric
from ..utils import get_layer_metric_array


@metric('synflow', bn=False, mode='param')
@metric('synflow_bn', bn=True, mode='param')
def compute_synflow_per_weight(net, inputs, targets, mode, split_data=1, loss_fn=None):

    device = inputs.device

    #convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    #convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    # keep signs of all params
    signs = linearize(net)
    
    # Compute gradients with input of 1s 
    net.zero_grad()
    net.double()
    input_dim = list(inputs[0,:].shape)
    inputs = torch.ones([1] + input_dim, requires_grad=True).double().to(device)
    output = net.forward(inputs)
    if type(output) is tuple:
        # (feat, logits) = output
        _, output = output
    torch.sum(output.double()).backward() 

    # select the gradients that we want to use for search/prune
    def synflow(layer):
        if layer.weight.grad is not None:
            #print(torch.nonzero(layer.weight.grad, as_tuple=False), layer)
            return torch.abs(layer.weight * layer.weight.grad)
        else:
            return torch.zeros_like(layer.weight)
    
    grads_abs = get_layer_metric_array(net, synflow, mode)

    # apply signs of all params
    nonlinearize(net, signs)

    return grads_abs
