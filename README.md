# Skeleton-CutMix

This is the repository for the paper **Skeleton-CutMix: Mixing Up Skeleton with Probabilistic Bone Exchange for Supervised Domain Adaptation**.

You may need to manually change paths for datasets in some places. 

### Cross-dataset setting
We use [NTU-60](https://rose1.ntu.edu.sg/dataset/actionRecognition/) and [ETRI-Activity3D](https://nanum.etri.re.kr/share/judekim/HMI?lang=En_us) for the cross-dataset setting. We use 23 action pairs. We evaluate on both NTU $\rightarrow$ ETRI
and ETRI $\rightarrow$ NTU settings.

#### Data preparation
Get skeleton data for both datasets, and follow the `readme.txt` in `data_gen` to generate data files. Then you will have `*.npy` data files and `*.pkl` label files
required by `.yaml` experiment config files.

#### Experiments
**Run baseline (S+T)**

`sh rev/common23_new_base.sh <device> <config_idx>`

**Run Skeleton-CutMix-S**

`sh rev/common23_new_beta.sh <device> <config_idx>`

**Run Skeleton-CutMix-W**

Prepare weight matrix for part importance sampling (PIS). Follow `rev/prepare_pis.txt` to run model for each body part, gather classification accuracy scores, 
and calculate the weight. You may need to try different $T$ depending on different source domains. Or you can directly use our precomputed weight defined in `skcutmix_pis_new.py`


Then `sh rev/common23_new_pis.sh <device> <T>`




### Acknowledgement
Some of the codes are borrowed from [ST-GCN](https://github.com/yysijie/st-gcn/), [2s-AGCN](https://github.com/lshiwjx/2s-AGCN), [DAGE](https://github.com/LukasHedegaard/dage). Thanks for their great work!

### Contact 
Hanchao Liu [liuhc21 at mails.tsinghua.edu.cn]



