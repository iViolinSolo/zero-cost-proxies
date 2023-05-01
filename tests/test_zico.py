def test_tcet():
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
    # net = models.resnet50()  # BUG fix: use resnet50 instead, because densenet161 using F.adaptive_avg_pool2d instead of nn.AdaptiveAvgPool2d, we cannot test it on zen.
    net.to(device)

    # setup dataloader
    from alethiometer import get_cifar_dataloaders
    train_loader, test_loader = get_cifar_dataloaders(32, 64, 'cifar10', 2)

    from alethiometer import calc_zc_metrics
    mts = [
        'zico',
    ]
    # mts = ['tenas', 'ntk']
    results = calc_zc_metrics(metrics=mts, model=net, train_queue=train_loader, device=device, aggregate=True)
    print(results)
    


if __name__ == '__main__':
    test_tcet()
