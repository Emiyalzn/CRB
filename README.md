# CRB

Official implementation for our paper:

[On Collective Robustness of Bagging against Data Poisoning](https://arxiv.org/abs/2205.13176)

[Ruoxin Chen](https://scholar.google.com/citations?user=kNky8dEAAAAJ&hl=en&oi=ao), [Zenan Li](https://scholar.google.com/citations?user=D0FZhYoAAAAJ&hl=en&oi=ao), [Jie Li*](https://scholar.google.com/citations?user=Krl5HRcAAAAJ&hl=en), [Chentao Wu](https://scholar.google.com/citations?user=RAs-wnEAAAAJ&hl=en&oi=ao), [Junchi Yan](https://scholar.google.com/citations?user=ga230VoAAAAJ&hl=en&oi=ao)

Shanghai Jiaotong University

In ICML 2022 (* correspondence).

![hashbagging](imgs/subtrainset.png)

## Environment

For the environment, We use python3.7, PyTorch, and torchvision, and assume CUDA capabilities. In addition, we need `gurobipy` since we formulate the certification process as a BILP problem.

## Getting Started

The code mainly contains two directories: `certify` (in which we run gurobi to calculate the collective robustness and certification accuracy) and `partition` (in which we pretrain models and get predictions).

We implement the certification for Vanilla and Hash Bagging (as mentioned in the paper) separately. Specifically, we provide an example for running our code. Assume we want to calculate the collective robustness for 50 classifiers (each with 2% of training samples) using Hash Bagging on Cifar-10:

- First, we create the partition file for training:

  ```
  python ./partition/hash/cv/partition_data_norm_hash.py --dataset cifar --portion 0.02 --partitions 50
  ```

  This will create `partitions_hash_mean_cifar_50_0.02.pth` under `partition/hash/cv`.

- Then, we train the subclassifiers and make predictions on the test set:

  ```
  python ./partition/hash/cv/train_cifar_nin_hash.py --num_partitions 50 --start_partition 0 --num_partitions_range 50 --portion 0.02
  python ./partition/hash/cv/evaluate_cifar_nin_hash.py --models cifar_nin_hash_partitions_50_portion_0.02
  ```

  This will create `cifar_nin_hash_partitions_50_portion_0.02.pth`  under `partition/hash/cv/evaluations`.

- Move the evaluation file under `./certify/evaluations` (create the directory beforehand), then we can run the certification:

  ```
  python ./certify/main_cv_hash.py rob cifar 50 --portion 0.02 --num_poison xx --scale xx 
  ```

  This will give the collective robustness we want based on Gurobi.

CV datasets can be downloaded automatically in our codes. However, for the classic datasets, you need to download the .csv files online artificially (the urls have been listed in the paper).

## Citation

If you use this code for your research, please cite our [paper](https://arxiv.org/abs/2205.13176):

```shell
@InProceedings{pmlr-v162-chen22k,
  title = 	 {On Collective Robustness of Bagging Against Data Poisoning},
  author =       {Chen, Ruoxin and Li, Zenan and Li, Jie and Yan, Junchi and Wu, Chentao},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {3299--3319},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/chen22k/chen22k.pdf},
  url = 	 {https://proceedings.mlr.press/v162/chen22k.html},
}
```

We also thank two papers here for their inspiration to our work.

```shell
@inproceedings{DBLP:conf/iclr/SchuchardtBKG21,
  author    = {Jan Schuchardt and
               Aleksandar Bojchevski and
               Johannes Klicpera and
               Stephan G{\"{u}}nnemann},
  title     = {Collective Robustness Certificates: Exploiting Interdependence in
               Graph Neural Networks},
  booktitle = {9th International Conference on Learning Representations, {ICLR} 2021,
               Virtual Event, Austria, May 3-7, 2021},
  publisher = {OpenReview.net},
  year      = {2021},
  url       = {https://openreview.net/forum?id=ULQdiUTHe3y},
  timestamp = {Wed, 23 Jun 2021 17:36:39 +0200},
  biburl    = {https://dblp.org/rec/conf/iclr/SchuchardtBKG21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

```shell
@inproceedings{DBLP:conf/iclr/0001F21,
  author    = {Alexander Levine and
               Soheil Feizi},
  title     = {Deep Partition Aggregation: Provable Defenses against General Poisoning
               Attacks},
  booktitle = {9th International Conference on Learning Representations, {ICLR} 2021,
               Virtual Event, Austria, May 3-7, 2021},
  publisher = {OpenReview.net},
  year      = {2021},
  url       = {https://openreview.net/forum?id=YUGG2tFuPM},
  timestamp = {Wed, 23 Jun 2021 17:36:39 +0200},
  biburl    = {https://dblp.org/rec/conf/iclr/0001F21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

