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
nnet_dir=exp/xvector_dgr5
train_cmd=run.pl
stage=0
rm -rf $nnet_dir/xvectors_enrollments/{utt2spk,xvector.scp}
if [ $stage -le 0 ]; then
for dir in $nnet_dir/xvectors_enrollments/*; do
  base=`basename $dir`
  if [ -d data/enrollments/$base ]; then
    cat data/enrollments/$base/utt2spk >> $nnet_dir/xvectors_enrollments/utt2spk
    utils/utt2spk_to_spk2utt.pl $nnet_dir/xvectors_enrollments/utt2spk > $nnet_dir/xvectors_enrollments/spk2utt
    cat $nnet_dir/xvectors_enrollments/$base/xvector.scp >> $nnet_dir/xvectors_enrollments/xvector.scp
  fi
done
fi

if [ $stage -le 1 ]; then
ivector-mean ark:$nnet_dir/xvectors_enrollments/spk2utt scp:$nnet_dir/xvectors_enrollments/xvector.scp \
  ark,scp:$PWD/$nnet_dir/xvectors_enrollments/spk_xvector.ark,$PWD/$nnet_dir/xvectors_enrollments/spk_xvector.scp ark,t:$PWD/$nnet_dir/xvectors_enrollments/spk_num_utts.ark 2> log
fi

