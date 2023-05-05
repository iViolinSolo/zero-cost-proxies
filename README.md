<!--
 * @Author: ViolinSolo
 * @Date: 2023-03-26 10:11:01
 * @LastEditTime: 2023-05-05 17:15:45
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
=      Your network need have attribute fn:             =
=         `forward_before_global_avg_pool(inputs)`      =
=      to calculate zenas score                         =
=      (see sample code in tests/test_zc.py)            =
=-------------------------------------------------------=
= tenas,                                                =
=      must work in `gpu` env,                          =
=      might encouter bug on `cpu`.                     =
=      also contains metrics:                           =
= ntk,                                                  =
= lrn,                                                  = 
=-------------------------------------------------------=
= zico, not work in torch-cpu, I will check it later.   =
=     zico must use at least two batches of data,       =
=     in order to calculate cross-batch (non-zero) std  =
=-------------------------------------------------------=
= tcet,                                                 =
=========================================================
```

## 0. How to install.
1. First create conda env with python version >= 3.6, this repo has been completely tested on python 3.9.  
    ```bash
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
    ```
2. Install torch, torchvision, cudatoolkit.
    ```bash
    Tested on:
    pytorch==1.13.1  (py3.9_cuda11.6_cudnn8.3.2_0)
    python==3.9.16
    cuda 11.6
    torchvision==0.14.1 (py39_cu116)
    torchaudio==0.13.1 (py39_cu116)
    ```
3. this repo is perfectly compatible with current mainstream zc testing framework, including zennas, naslib, nb201 related repos, nb101, nb1shot1, blox, etc.  

4. If you still cannot use this repo, try to contact me, or try to setup some mainstream nas testing benchmarks, then most problems would be solved.  

5. Finally, if all the previous basic enviroment requirements are met, then try this lib with just one single command.
    ```
    pip install -e .
    # running this command under the root directory where the setup.py locates in.
    ```
6. check installation success.
    ```bash
    cd tests/
    python test_zc.py
    ```



## 1. Tests
ImageNet16-120 cannot be automatically downloaded. Using script under `scripts/download_data.sh` to download:
```bash
source scripts/download_data.sh nb201 ImageNet16-120
# do not use `bash`, use `source` instead
```
## 2. Versions
- V1.1.1  
Fix warnings in `tenas`, now using new torch api to calc eigenvalue.  
Fix bug in `tcet`, add net instance deep copy to avoid weight changes, add manually designed `tcet` copy process, remove bn in synflow, add bn in snip.
- V1.1.0  
Add `tcet` metric, which calculates TCET score.
Add `snr` metrics, which calculates SNR family scores.
- V1.0.10  
add `zico` metric, which calculates ZICO score.  
- V1.0.9  
fix readme format, no code change.
- V1.0.8  
fix bug in `nwot_relu` for wrong for/backward fn register,  
fix bug in `zen` for missed necessary attribute check, add test sample for `zen` metric,  
fix bug in `zen` for return value have not .item() attribute,  
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


## 3. Quick Bug Fix
1. if you encouther this error:   
    `RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'`
    ```bash
    Traceback (most recent call last):
    File "/home/u2280887/GitHub/zero-cost-proxies/tests/test_zc.py", line 87, in <module>
        test_zc_proxies()
    File "/home/u2280887/GitHub/zero-cost-proxies/tests/test_zc.py", line 49, in test_zc_proxies
        results = calc_zc_metrics(metrics=mts, model=net, train_queue=train_loader, device=device, aggregate=True)
    File "/home/u2280887/GitHub/zero-cost-proxies/alethiometer/zc_proxy.py", line 115, in calc_zc_metrics
        mt_vals = calc_vals(net_orig=model, trainloader=train_queue, device=device, metric_names=metrics, loss_fn=loss_fn)
    File "/home/u2280887/GitHub/zero-cost-proxies/alethiometer/zc_proxy.py", line 101, in calc_vals
        raise e
    File "/home/u2280887/GitHub/zero-cost-proxies/alethiometer/zc_proxy.py", line 73, in calc_vals
        val = M.calc_metric(mt_name, net_orig, device, inputs, targets, loss_fn=loss_fn, split_data=ds)
    File "/home/u2280887/GitHub/zero-cost-proxies/alethiometer/zero_cost_metrics/__init__.py", line 42, in calc_metric
        return _metric_impls[name](net, device, *args, **kwargs)
    File "/home/u2280887/GitHub/zero-cost-proxies/alethiometer/zero_cost_metrics/__init__.py", line 24, in metric_impl
        ret = func(net, *args, **kwargs, **impl_args)
    File "/home/u2280887/GitHub/zero-cost-proxies/alethiometer/zero_cost_metrics/tenas.py", line 316, in compute_TENAS_score
        RN = compute_RN_score(net, inputs, targets, split_data, loss_fn, num_batch)
    File "/home/u2280887/GitHub/zero-cost-proxies/alethiometer/zero_cost_metrics/tenas.py", line 201, in compute_RN_score
        num_linear_regions = float(lrc_model.forward_batch_sample()[0])
    File "/home/u2280887/GitHub/zero-cost-proxies/alethiometer/zero_cost_metrics/tenas.py", line 170, in forward_batch_sample
        return [LRCount.getLinearReginCount() for LRCount in self.LRCounts]
    File "/home/u2280887/GitHub/zero-cost-proxies/alethiometer/zero_cost_metrics/tenas.py", line 170, in <listcomp>
        return [LRCount.getLinearReginCount() for LRCount in self.LRCounts]
    File "/home/u2280887/GitHub/zero-cost-proxies/alethiometer/zero_cost_metrics/tenas.py", line 93, in getLinearReginCount
        self.calc_LR()
    File "/home/u2280887/miniconda3/envs/zc-alth/lib/python3.9/site-packages/torch/autograd/grad_mode.py", line 27, in decorate_context
        return func(*args, **kwargs)
    File "/home/u2280887/GitHub/zero-cost-proxies/alethiometer/zero_cost_metrics/tenas.py", line 62, in calc_LR
        res = torch.matmul(self.activations.half(), (1-self.activations).T.half())
    RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'
    ```
    please check your lib installation, we need gpu support for `torch.half()`, please check your cuda version and pytorch version, and reinstall pytorch with cuda support. It seem current cpu version of pytorch does not support `torch.half()`, even if we are using float32 not float16.

2. ....