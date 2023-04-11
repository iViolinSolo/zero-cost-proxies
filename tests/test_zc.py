'''
Author: ViolinSolo
Date: 2023-04-07 19:25:08
LastEditTime: 2023-04-11 13:09:09
LastEditors: ViolinSolo
Description: Test zc-proxies.
FilePath: /zero-cost-proxies/tests/test_zc.py
'''

def test_dataloader():
    """
    test on dataloader.
    """
    from alethiometer.datasets import get_cifar_dataloaders

    train_dataloader, test_dataloader = get_cifar_dataloaders(64, 128, 'cifar10', 2, skip_download_check=False)
    train_dataloader, test_dataloader = get_cifar_dataloaders(64, 128, 'cifar100', 2, skip_download_check=False)
    train_dataloader, test_dataloader = get_cifar_dataloaders(64, 128, 'ImageNet16-120', 2, skip_download_check=False)

# end def

def test_zc_proxies():
    import torch
    import random
    import numpy as np
    import torchvision.models as models

    seed = 42

    # REPRODUCIBILITY
    torch.manual_seed(seed) #use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)  #For custom operators, you might need to set python seed as well
    np.random.seed(seed)  #If you or any of the libraries you are using rely on NumPy, you can seed the global NumPy RNG with

    # setup network
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = models.densenet161()
    net.to(device)

    from alethiometer.zc_proxy import calc_zc_metrics

if __name__ == "__main__":
    test_dataloader()