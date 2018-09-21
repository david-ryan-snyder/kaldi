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
spkr=$1
train_cmd=run.pl

name=enrollments
stage=0
utt=${spkr}_$(date +%Y%m%d%H%M%S)

if [ $stage -le 0 ]; then
  mkdir -p data/$name/$utt
  echo "$spkr $utt" > data/$name/$utt/spk2utt
  echo "$utt $spkr" > data/$name/$utt/utt2spk
  utils/fix_data_dir.sh data/$name/$utt &> log
  mkdir -p $nnet_dir/xvectors_enrollments/$utt/
  copy-vector scp:$nnet_dir/xvectors_test/xvector.scp ark,scp:$PWD/$nnet_dir/xvectors_enrollments/$utt/xvector.ark,$PWD/$nnet_dir/xvectors_enrollments/$utt/xvector.scp &> log
  awk -v utt=$utt '{print utt, $2}' $PWD/$nnet_dir/xvectors_enrollments/$utt/xvector.scp > $PWD/$nnet_dir/xvectors_enrollments/$utt/xvector.scp.new
  mv $PWD/$nnet_dir/xvectors_enrollments/$utt/xvector.scp.new $PWD/$nnet_dir/xvectors_enrollments/$utt/xvector.scp
  echo "DONE"
fi
