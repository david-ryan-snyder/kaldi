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
spkr=$2
s=$3
e=$4
train_cmd=run.pl

name=enrollments
stage=0
utt=${spkr}_$(date +%Y%m%d%H%M%S)

if [ $stage -le 0 ]; then
  rm -rf $utt.{m4a,wav}
  rm -rf data/enrollments/$utt
  f=$utt.m4a
  rm -f $f
  if [ -z "$s" ]; then
    s=0
    e=1000
  fi
  ffmpeg -ss $s -i $(youtube-dl -x -f 140 --get-url $ytid) -t $e -c:a copy $utt.m4a &> log
  reco=`basename $f` &> log
  utt="${reco/%.m4a/}" &> log
  ffmpeg -y -i "$f" -ac 1 -ar 16000 "${f/%m4a/wav}" &> /dev/null
  python local/make_enroll_data_single.py "${f/%m4a/wav}" data/${name}/$utt $spkr &> log
  #utils/fix_data_dir.sh data/${name} &> log
fi

if [ $stage -le 1 ]; then
  echo "extracting embedding for utterance $utt ..." &> log
  create_embedding_internal.sh $name/$utt $nnet_dir &> log
  rm $utt.*
  echo "DONE"
fi
