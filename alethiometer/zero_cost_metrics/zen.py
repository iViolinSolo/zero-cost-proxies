'''
Author: ViolinSolo
Date: 2023-04-23 12:57:36
LastEditTime: 2023-04-28 22:42:23
LastEditors: ViolinSolo
Description: zen
FilePath: /zero-cost-proxies/alethiometer/zero_cost_metrics/zen.py
'''

# =============================================================================
#   Copyright (C) 2010-2021 Alibaba Group Holding Limited.
#  https://github.com/idstcv/ZenNAS/blob/main/ZeroShotProxy/compute_zen_score.py
# =============================================================================

import torch
import numpy as np

from torch import nn
from . import metric


def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is None:
                    continue
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net


@metric("zen", bn=True)
def compute_zen_score(net, inputs, targets, loss_fn=None, split_data=1,
                      repeat=1, mixup_gamma=1e-2, fp16=False):
    nas_score_list = []

    device = inputs.device
    dtype = torch.half if fp16 else torch.float32

    with torch.no_grad():
        for repeat_count in range(repeat):
            network_weight_gaussian_init(net)
            input = torch.randn(size=list(inputs.shape), device=device, dtype=dtype)
            input2 = torch.randn(size=list(inputs.shape), device=device, dtype=dtype)
            mixup_input = input + mixup_gamma * input2

            if not hasattr(net, 'forward_before_global_avg_pool'):
                suggestion_msg = "\n\tPlease implement forward_before_global_avg_pool() in your network.\n\tYou can follow implementation of fn:`forward_pre_GAP()` in ZenNet:\n\thttps://github.com/idstcv/ZenNAS/blob/2629dc5692b3d9d01ef94b559e6bd4a4b114b617/Masternet.py#L98"
                raise NotImplementedError(suggestion_msg)

            output = net.forward_before_global_avg_pool(input)
            mixup_output = net.forward_before_global_avg_pool(mixup_input)

            nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
            nas_score = torch.mean(nas_score)

            # compute BN scaling
            log_bn_scaling_factor = 0.0
            for m in net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                    log_bn_scaling_factor += torch.log(bn_scaling_factor)
                pass
            pass
            nas_score = torch.log(nas_score) + log_bn_scaling_factor
            nas_score_list.append(float(nas_score))

    ## 95% confidence interval of NAS score
    # std_nas_score = np.std(nas_score_list)
    # avg_precision = float(1.96 * std_nas_score / np.sqrt(len(nas_score_list)))
    # avg_nas_score = float(np.mean(nas_score_list))
    avg_nas_score = np.mean(nas_score_list)  # fix bug when calling .item() on this return value

    return avg_nas_score