#! /bin/bash
# -*- coding: utf-8 -*-

# This script is modified from https://github.com/bigpon/QPPWG.

trainset=arctic_train_16kHz  # training set
validset=arctic_valid_16kHz  # validation set
evalset=arctic_eval_16kHz    # evaluation set
gpu=0           # gpu id
conf=uSFGAN_40  # name of config 
resume=None     # number of iteration of resume model
iter=400000     # number of iteration of testing model
scaled=0.50     # scaled ratio of f0
stage=          # running stage (0-3)
                # stage 0: Preprocessing
                # stage 1: uSFGAN training
                # stage 2: uSFGAN decoding (analysis-synthesis)
                # stage 3: uSFGAN decoding (scaled F0)
. ../parse_options.sh || exit 1;

export LD_LIBRARY_PATH=''
export CUDA_HOME=''
export CUDA_DEVICE_ORDER=''

# Preprocessing
if echo ${stage} | grep -q 0; then
    echo "Preprocessing."
    python run.py -C ${conf} -T ${trainset} -V ${validset} -E ${evalset} -0
fi

# uSFGAN training
if echo ${stage} | grep -q 1; then
    echo "uSFGAN training."
    python run.py -g ${gpu} -C ${conf} \
    -T ${trainset} -V ${validset} -R ${resume} -1
fi

# uSFGAN decoding w/ natural acoustic features
if echo ${stage} | grep -q 2; then
    echo "uSFGAN decoding (natural)."
    python run.py -g ${gpu} -C ${conf} \
    -T ${trainset} -E ${evalset} -I ${iter} -2
fi

# uSFGAN decoding w/ scaled F0
if echo ${stage} | grep -q 3; then
    echo "uSFGAN decoding ( ${scaled} x F0)."
    python run.py -g ${gpu} -C ${conf} -f ${scaled}\
    -T ${trainset} -E ${evalset} -I ${iter} -2
fi