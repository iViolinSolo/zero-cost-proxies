'''
Author: ViolinSolo
Date: 2023-04-07 19:25:08
LastEditTime: 2023-04-29 10:45:56
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
    # net = models.densenet161()
    net = models.resnet50()  # BUG fix: use resnet50 instead, because densenet161 using F.adaptive_avg_pool2d instead of nn.AdaptiveAvgPool2d, we cannot test it on zen.
    net.to(device)

    # setup dataloader
    from alethiometer import get_cifar_dataloaders
    train_loader, test_loader = get_cifar_dataloaders(32, 32, 'cifar10', 2)

    from alethiometer import calc_zc_metrics
    mts = ['snip', 'grasp', 'grad_norm', 'synflow', 'nwot', 'lnwot', 'nwot_relu', 'tenas', 'lrn', 'ntk']
    # mts = ['tenas', 'ntk']
    results = calc_zc_metrics(metrics=mts, model=net, train_queue=train_loader, device=device, aggregate=True)
    print(results)
    

    # ==================== test zen ====================
    # test zen
    print('-'*20)
    print('test zen .....')
    import types
    def forward_pre_GAP(self, x: torch.Tensor) -> list:
        outputs = []

        def hook_fn(module, inputs, output_t):
            # print(f'Input tensor shape: {inputs[0].shape}')
            # print(f'Output tensor shape: {output_t.shape}')
            outputs.append(inputs[0])

        for m in self.modules():
            # print('m.type: ', type(m))
            if isinstance(m, torch.nn.AdaptiveAvgPool2d):
                m.register_forward_hook(hook_fn)

        # self.forward(x, None)
        self.forward(x)

        assert len(outputs) == 1
        return outputs[0]
    
    if not hasattr(net, 'forward_before_global_avg_pool'):
        net.forward_before_global_avg_pool = types.MethodType(forward_pre_GAP, net)

    results = calc_zc_metrics(metrics=['zen'], model=net, train_queue=train_loader, device=device, aggregate=True)
    print(results)

    # ==================== test zico ====================
    # test zico
    print('-'*20)
    print('test zico .....')

    results = calc_zc_metrics(metrics=['zico'], model=net, train_queue=train_loader, device=device, aggregate=True)
    print(results)


if __name__ == "__main__":
    test_dataloader()
    test_zc_proxies()
