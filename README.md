
# Unified Source-Filter GAN (uSFGAN)

<b>I released a new PyTorch implementation of uSFGAN in addition to a better model, Harmonic-plus-Noise uSFGAN [here](https://github.com/chomeyama/HN-UnifiedSourceFilterGAN).</b>

This is official PyTorch implementation of [uSFGAN](https://arxiv.org/abs/2104.04668), which is a unified source-filter network based on factorization of [QPPWG](https://github.com/bigpon/QPPWG) by Yi-Chiao Wu @ Nagoya University ([@bigpon](https://github.com/bigpon)).

<p align="center">
<img width="754" alt="architecture" src="https://user-images.githubusercontent.com/49127218/121571723-55ca7400-ca5e-11eb-8c17-b93aeaf617fc.png">
</p>

In this repo, we provide an example to train and test uSFGAN as a vocoder for [WORLD](https://doi.org/10.1587/transinf.2015EDP7457) acoustic features.
More details can be found on our [Demo](https://chomeyama.github.io/UnifiedSourceFilterGAN-Demo/) page.

## Requirements

This repository is tested on Ubuntu 20.04 with a Titan RTX 3090 GPU.

- Python 3.8+
- Cuda 11.0
- CuDNN 7+
- PyTorch 1.7.1+


## Environment setup

```bash
$ cd UnifiedSourceFilterGAN
$ pip install -e .
```

Please refer to the [PWG](https://github.com/kan-bayashi/ParallelWaveGAN) repo for more details.

## Folder architecture
- **egs**:
The folder for projects.
- **egs/vcc18**:
The folder of the VCC2018 project.
- **egs/vcc18/exp**:
The folder for trained models.
- **egs/vcc18/conf**:
The folder for configs.
- **egs/vcc18/data**:
The folder for corpus related files (wav, feature, list ...).
- **usfgan**:
The folder of the source codes.

Projects on [CMU-ARCTIC](http://www.festvox.org/cmu_arctic/) corpus are also available
- Check **egs/arctic/***
- Dataset separation is based on [Official NSF implementation](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts)

## Run

### Corpus and path setup

- Modify the corresponding CUDA paths in `egs/vcc18/run.py`.
- Download the [Voice Conversion Challenge 2018](https://datashare.is.ed.ac.uk/handle/10283/3061) (VCC2018) corpus to run the uSFGAN example.

```bash
$ cd egs/vcc18
# Download training and validation corpus
$ wget -o train.log -O data/wav/train.zip https://datashare.is.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_training.zip
# Download evaluation corpus
$ wget -o eval.log -O data/wav/eval.zip https://datashare.is.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_evaluation.zip
# unzip corpus
$ unzip data/wav/train.zip -d data/wav/
$ unzip data/wav/eval.zip -d data/wav/
```

- **Training wav lists**: `data/scp/vcc18_train_22kHz.scp`.
- **Validation wav lists**: `data/scp/vcc18_valid_22kHz.scp`.
- **Testing wav list**: `data/scp/vcc18_eval_22kHz.scp`.

### Preprocessing

```bash
# Extract WORLD acoustic features and statistics of training and testing data
$ bash run.sh --stage 0 --conf uSFGAN_60
```

- WORLD-related settings can be changed in `egs/vcc18/conf/vcc18.uSFGAN_60.yaml`.
- If you want to use another corpus, please create a corresponding config and a file including power thresholds and f0 ranges like `egs/vcc18/data/pow_f0_dict.yml`.
- More details about feature extraction can be found in the [QPNet](https://github.com/bigpon/QPNet) repo.
- The lists of auxiliary features will be automatically generated.
- **Training aux lists**: `data/scp/vcc18_train_22kHz.list`.
- **Validation aux lists**: `data/scp/vcc18_valid_22kHz.list`.
- **Testing aux list**: `data/scp/vcc18_eval_22kHz.list`.


### uSFGAN training

```bash
# Training a uSFGAN model with the 'uSFGAN_60' config and the 'vcc18_train_22kHz' and 'vcc18_valid_22kHz' sets.
$ bash run.sh --gpu 0 --stage 1 --conf uSFGAN_60 \
--trainset vcc18_train_22kHz --validset vcc18_valid_22kHz
```

- The gpu ID can be set by --gpu GPU_ID (default: 0)
- The model architecture can be set by --conf CONFIG (default: uSFGAN_60)
- The trained model resume can be set by --resume NUM (default: None)


### uSFGAN testing

```bash
# uSFGAN/QPPWG/PWG decoding w/ natural acoustic features
$ bash run.sh --gpu 0 --stage 2 --conf uSFGAN_60 \
--iter 400000 --trainset vcc18_train_22kHz --evalset vcc18_eval_22kHz
# uSFGAN/QPPWG/PWG decoding w/ scaled f0 (ex: halved f0).
$ bash run.sh --gpu 0 --stage 3 --conf uSFGAN_60 --scaled 0.50 \
--iter 400000 --trainset vcc18_train_22kHz --evalset vcc18_eval_22kHz
```

### Monitor training progress

```bash
$ tensorboard --logdir exp
```

- The training time of uSFGAN_60 with a TITAN RTX 3090 is around 6 days.

## Citation
If you find the code is helpful, please cite the following article.

```
@inproceedings{yoneyama21_interspeech,
  author={Reo Yoneyama and Yi-Chiao Wu and Tomoki Toda},
  title={{Unified Source-Filter GAN: Unified Source-Filter Network Based On Factorization of Quasi-Periodic Parallel WaveGAN}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={2187--2191},
  doi={10.21437/Interspeech.2021-517}
}
```

## Authors

Development:
Reo Yoneyama @ Nagoya University ([@chomeyama](https://github.com/chomeyama))<br>
E-mail: `yoneyama.reo@g.sp.m.is.nagoya-u.ac.jp`

Advisor:
Yi-Chiao Wu @ Nagoya University ([@bigpon](https://github.com/bigpon))<br>
E-mail: `yichiao.wu@g.sp.m.is.nagoya-u.ac.jp`

Tomoki Toda @ Nagoya University<br>
E-mail: `tomoki@icts.nagoya-u.ac.jp`
