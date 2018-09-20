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
#See http://kaldi-asr.org/models.html and
# https://david-ryan-snyder.github.io/2017/10/04/model_sre16_v2.html
# for details.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

nnet_dir=exp/xvector_dgr5/
ytid=$1
s=$2
e=$3
train_cmd=run.pl

name=test
stage=0

if [ $stage -le 0 ]; then
  rm -rf data/$name
  rm -rf foo.m4a
  rm -rf foo.wav
  f=foo.m4a
  rm -f $f
  if [ -z "$s" ]; then
    s=0
    e=1000
  fi
  ffmpeg -ss $s -i $(youtube-dl -x -f 140 --get-url $ytid) -t $e -c:a copy foo.m4a &> log
  reco=`basename $f` &> log
  utt="${reco/%.m4a/}" &> log
  ffmpeg -y -i "$f" -ac 1 -ar 16000 "${f/%m4a/wav}" &> /dev/null
  python local/make_test_data_single.py "${f/%m4a/wav}" data/${name} &> log
  utils/fix_data_dir.sh data/${name} &> log
fi

if [ $stage -le 1 ]; then
  echo "extracting embedding for utterance $utt ..." &> log
  create_embedding_internal.sh $name $nnet_dir &> log
  echo "DONE"
fi
