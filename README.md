<!--
 * @Author: ViolinSolo
 * @Date: 2023-03-26 10:11:01
 * @LastEditTime: 2023-04-28 21:48:00
 * @LastEditors: ViolinSolo
 * @Description: Readme
 * @FilePath: /zero-cost-proxies/README.md
-->
# zero-cost-proxies
Independent ZC proxies only for testing on it. 

Modified and simplified from [foresight repo](https://github.com/SamsungLabs/zero-cost-nas), fix some bugs in model output, remove some unwanted code snippets.

Supported zc-metrics are:
```
=========================================================
= grad_norm,                                            =
=-------------------------------------------------------=
= grasp,                                                =
=-------------------------------------------------------=
= snip,                                                 =
=-------------------------------------------------------=
= synflow,                                              =
=-------------------------------------------------------=
= nwot, (NASWOT)                                        =
=       [nwot, nwot_Kmats]                              =
=-------------------------------------------------------=
= lnwot, (Layerwise NASWOT)                             =
=       [lnwot, lnwot_Kmats]                            =
=-------------------------------------------------------=
= nwot_relu, (original RELU based NASWOT metric)        =
=       [nwot_relu, nwot_relu_Kmats]                    =
=-------------------------------------------------------=
= zen,                                                  =
=========================================================
```


## 1. Tests
ImageNet16-120 cannot be automatically downloaded. Using script under `scripts/download_data.sh` to download:
```bash
source scripts/download_data.sh nb201 ImageNet16-120
# do not use `bash`, use `source` instead
```
## 2. Versions
- V1.0.8
fix bug in `nwot_relu` for wrong for/backward fn register,
add `tenas` metric, which calculates TE-NAS score. (`tenas`, `ntk`, `lrn`)
- V1.0.7
add `zen` metric, which calculates ZenNAS score.
- V1.0.6  
add original `naswot` implements based on RELU, can be calculated using metirc `nwot_relu`, also fix potential oom bug, and more reliable GPU memory cache removal code snippets.  
- V1.0.5  
add `naswot, lnwot` into mats
- V1.0.4  
fix bugs in calculation, add more test codes.
- V1.0.3  
add shortcuts to import directly from package root directory.

