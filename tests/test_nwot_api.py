'''
Author: ViolinSolo
Date: 2023-04-25 09:54:56
LastEditTime: 2023-04-25 10:05:55
LastEditors: ViolinSolo
Description: test on nwot lnwot and its corresponding return_mat api
FilePath: /zero-cost-proxies/tests/test_nwot_api.py
'''

def test_nwot_api():
    """
    test on nwot lnwot and its corresponding return_mat api
    """
    from alethiometer import calc_zc_metrics, get_cifar_dataloaders
    import torch

    # get dataloader
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_cifar_dataloaders(64, 64, 'cifar10', 2)

    # get model
    from torchvision.models import resnet18
    net = resnet18()
    net.to(device)

    # get metrics
    mts = ['nwot', 'lnwot', 'nwot_Kmats', 'lnwot_Kmats']
    results = calc_zc_metrics(metrics=mts, model=net, train_queue=train_loader, device=device, aggregate=False)

    print('nwot:', type(results['nwot']), results['nwot'])
    print('lnwot:', type(results['lnwot']), results['lnwot'])
    print('nwot_Kmats:', type(results['nwot_Kmats']), results['nwot_Kmats'])
    print('lnwot_Kmats:', type(results['lnwot_Kmats']), results['lnwot_Kmats'])
    
# end def

if __name__ == "__main__":
    test_nwot_api()