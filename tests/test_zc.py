'''
Author: ViolinSolo
Date: 2023-04-07 19:25:08
LastEditTime: 2023-04-11 12:56:57
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
    pass

if __name__ == "__main__":
    test_dataloader()
