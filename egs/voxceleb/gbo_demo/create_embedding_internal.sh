#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (mostly EERs) are inline in comments below.
#
# This example demonstrates a "bare bones" NIST SRE 2016 recipe using xvectors.
# In the future, we will add score-normalization and a more effective form of
# PLDA domain adaptation.
#
# Pretrained models are available for this recipe.
# See http://kaldi-asr.org/models.html and
# https://david-ryan-snyder.github.io/2017/10/04/model_sre16_v2.html
# for details.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
train_cmd=run.pl
nnet_dir=$2
name=$1

utils/fix_data_dir.sh data/${name} &> /dev/null
echo "MFCC";
time steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 1 --cmd "$train_cmd" \
data/${name} exp/make_mfcc $mfccdir &> /dev/null
utils/fix_data_dir.sh data/${name} &> /dev/null
echo "VAD";
time sid/compute_vad_decision.sh --nj 1 --cmd "$train_cmd" \
data/${name} exp/make_vad $vaddir &> /dev/null
utils/fix_data_dir.sh data/${name} &> /dev/null
echo "extract xvectors";
time sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 1 \
$nnet_dir data/${name} \
$nnet_dir/xvectors_${name} &> /dev/null
