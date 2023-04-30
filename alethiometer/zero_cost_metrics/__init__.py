'''
Author: ViolinSolo
Date: 2023-04-06 17:43:34
LastEditTime: 2023-04-30 16:48:30
LastEditors: ViolinSolo
Description: init files 
FilePath: /zero-cost-proxies/alethiometer/zero_cost_metrics/__init__.py
'''

import gc
import torch

available_metrics = []
_metric_impls = {}

def metric(name, bn=True, copy_net=True, force_clean=True, **impl_args):
    def make_impl(func):
        def metric_impl(net_orig, device, *args, **kwargs):
            if copy_net:
                net = net_orig.get_prunable_copy(bn=bn).to(device)
            else:
                net = net_orig.to(device)
            try:
                ret = func(net, *args, **kwargs, **impl_args)
            finally:
                if copy_net and force_clean:
                    del net
                    torch.cuda.empty_cache()
                    gc.collect()
            return ret

        global _metric_impls
        if name in _metric_impls:
            raise KeyError(f'Duplicated metric! {name}')
        available_metrics.append(name)
        _metric_impls[name] = metric_impl
        return func
    return make_impl


def calc_metric(name, net, device, *args, **kwargs):
    return _metric_impls[name](net, device, *args, **kwargs)


# using the following import to trigger metric registration when using @metric
from . import snip
from . import synflow
from . import grasp
from . import grad_norm
from . import naswot # this is our NASWOT implementation (layer-wise or not, based on conv and fc)
from . import naswot_relu  # this is original RELU based NASWOT implementation
from . import zen 
from . import tenas # this is TENAS implementation, including lrn, ntk, tenas
from . import zico
from . import t_cet # this is our t_cet implementation
