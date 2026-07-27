# FLAME
This repository provides the official implementation of **[FLAME: Condensing Ensemble Diversity into a Single Network for Efficient Sequential Recommendation](https://doi.org/10.1145/3805712.3809560)**, accepted to the Full Papers Track of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (**SIGIR 2026**).

## 1. Overview
<img src="figures/flame.png" alt="Sample Figure">

## 2. Environment
```
conda env create --file env.yml
conda activate flame
```

## 3. Usage
```
python3 train.py --dataset amazon-beauty
```

## 4. Citation
```
@inproceedings{kim2026flame,
  title={FLAME: Condensing Ensemble Diversity into a Single Network for Efficient Sequential Recommendation},
  author={Kim, WooJoo and Kim, JunYoung and Lim, JaeHyung and Choi, SeongJin and Kang, SeongKu and Yu, HwanJo},
  booktitle={Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={823--833},
  year={2026}
}
```
