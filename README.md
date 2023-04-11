<!--
 * @Author: ViolinSolo
 * @Date: 2023-03-26 10:11:01
 * @LastEditTime: 2023-04-11 12:51:36
 * @LastEditors: ViolinSolo
 * @Description: Readme
 * @FilePath: /zero-cost-proxies/README.md
-->
# zero-cost-proxies
Independent ZC proxies only for testing on it. 

Modified and simplified from [foresight repo](https://github.com/SamsungLabs/zero-cost-nas), fix some bugs in model output, remove some unwanted code snippets.


ImageNet16-120 cannot be automatically downloaded. Using script under `scripts/download_data.sh` to download:
```bash
source scripts/download_data.sh
# do not use `bash`, use `source` instead
```