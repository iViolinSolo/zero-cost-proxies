<!--
 * @Author: ViolinSolo
 * @Date: 2023-03-26 10:11:01
 * @LastEditTime: 2023-04-11 13:55:39
 * @LastEditors: ViolinSolo
 * @Description: Readme
 * @FilePath: /zero-cost-proxies/README.md
-->
# zero-cost-proxies
Independent ZC proxies only for testing on it. 

Modified and simplified from [foresight repo](https://github.com/SamsungLabs/zero-cost-nas), fix some bugs in model output, remove some unwanted code snippets.

Supported zc-metrics are:
```
grad_norm,
grasp,
snip,
synflow,
```


## 1. Tests
ImageNet16-120 cannot be automatically downloaded. Using script under `scripts/download_data.sh` to download:
```bash
source scripts/download_data.sh nb201 ImageNet16-120
# do not use `bash`, use `source` instead
```
## 2. Versions
- V1.0.4  
fix bugs in calculation, add more test codes.
- V1.0.3  
add shortcuts to import directly from package root directory.

