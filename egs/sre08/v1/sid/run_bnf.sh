#!/bin/bash
# Copyright 2013   Daniel Povey
#           2014   David Snyder
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (EERs) are inline in comments below.

# This example script is still a bit of a mess, and needs to be
# cleaned up, but it shows you all the basic ingredients.

. cmd.sh
. path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
num_components=2048
trials_female=data/sre10_test_female/trials
trials_male=data/sre10_test_male/trials
trials=data/sre10_test/trials

# Extract speaker recogntion features.
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    data/train exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    data/sre exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    data/sre10_train exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    data/sre10_test exp/make_mfcc $mfccdir

# Extract DNN features. Note that this uses the hires MFCC config file.
steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj 40 \
    --cmd "$train_cmd" data/train_dnn exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj 40 \
    --cmd "$train_cmd" data/sre_dnn exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj 40 \
    --cmd "$train_cmd" data/sre10_train_dnn exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj 40 \
    --cmd "$train_cmd" data/sre10_test_dnn exp/make_mfcc $mfccdir

# Dump the BNF features. This assumes that you've created hires MFCCs
# in data/${dir}_dnn.
for dir in sre sre10_train sre10_test train; do
  utils/fix_data_dir.sh data/${dir}
  utils/fix_data_dir.sh data/${dir}_dnn
  sid/compute_bnf.sh \
  --cmd "$train_cmd -l mem_free=3G,ram_free=3G -l gpu=1" \
  --use-gpu yes \
  --chunk-size 500 \
  --nj 4 \
  --feat-type online \
  data/${dir}_dnn \
  data/${dir}_bnf exp/nnet exp/bnf exp/bnf
done

# Note: to see the proportion of voiced frames you can do,
# grep Prop exp/make_vad/vad_*.1.log

# Reduce the amount of training data for the UBM.
# Subset training data for faster sup-GMM initialization.
utils/subset_data_dir.sh data/train_bnf 32000 data/train_bnf_32k
utils/fix_data_dir.sh data/train_bnf_32k
utils/subset_data_dir.sh --utt-list data/train_bnf_32k/utt2spk data/train \
    data/train_32k
utils/fix_data_dir.sh data/train_32k

# Train UBM and i-vector extractor.
sid/train_diag_ubm_bnf.sh --cmd "$train_cmd -l mem_free=40G,ram_free=40G" \
    --nj 60 \
    --delta-order 1 \
    --norm-vars true \
    data/train_32k data/train_bnf_32k $num_components \
    exp/diag_ubm_$num_components

sid/train_full_ubm_bnf.sh --nj 60 --remove-low-count-gaussians false \
    --norm-vars true \
    --cmd "$train_cmd -l mem_free=25G,ram_free=25G" \
    data/train_32k \
    data/train_bnf_32k \
    exp/diag_ubm_$num_components exp/full_ubm_$num_components

sid/train_ivector_extractor_bnf.sh --cmd "$train_cmd -l mem_free=35G,ram_free=35G" \
  --norm-vars true \
  --ivector-dim 600 \
  --num-iters 5 exp/full_ubm_$num_components/final.ubm data/train data/train_bnf \
  exp/extractor

# Extract i-vectors.
sid/extract_ivectors_bnf.sh --cmd "$train_cmd -l mem_free=6G,ram_free=6G" --nj 50 \
   --norm-vars true \
   exp/extractor data/sre10_train \
   data/sre10_train_bnf \
   exp/ivectors_sre10_train

sid/extract_ivectors_bnf.sh --cmd "$train_cmd -l mem_free=6G,ram_free=6G" --nj 50 \
   --norm-vars true \
   exp/extractor data/sre10_test \
   data/sre10_test_bnf \
   exp/ivectors_sre10_test

sid/extract_ivectors_bnf.sh --cmd "$train_cmd -l mem_free=6G,ram_free=6G" --nj 50 \
   --norm-vars true \
   exp/extractor data/sre \
   data/sre_bnf \
   exp/ivectors_sre

local/scoring_common.sh data/sre data/sre10_train data/sre10_test \
  exp/ivectors_sre exp/ivectors_sre10_train \
  exp/ivectors_sre10_test

# Create a gender independent PLDA model and do scoring with the DNN system.
local/plda_scoring.sh data/sre data/sre10_train data/sre10_test \
  exp/ivectors_sre exp/ivectors_sre10_train \
  exp/ivectors_sre10_test $trials local/scores_ind_pooled
local/plda_scoring.sh --use-existing-models true data/sre data/sre10_train_female data/sre10_test_female \
  exp/ivectors_sre exp/ivectors_sre10_train_female \
  exp/ivectors_sre10_test_female $trials_female local/scores_ind_female
local/plda_scoring.sh --use-existing-models true data/sre data/sre10_train_male data/sre10_test_male \
  exp/ivectors_sre exp/ivectors_sre10_train_male \
  exp/ivectors_sre10_test_male $trials_male local/scores_ind_male

# Create gender dependent PLDA models and do scoring with the DNN system.
local/plda_scoring.sh data/sre_female data/sre10_train_female data/sre10_test_female \
  exp/ivectors_sre exp/ivectors_sre10_train_female \
  exp/ivectors_sre10_test_female $trials_female local/scores_dep_female
local/plda_scoring.sh data/sre_male data/sre10_train_male data/sre10_test_male \
  exp/ivectors_sre exp/ivectors_sre10_train_male \
   exp/ivectors_sre10_test_male $trials_male local/scores_dep_male
mkdir -p local/scores_dep_pooled
cat local/scores_dep_male/plda_scores local/scores_dep_female/plda_scores \
  > local/scores_dep_pooled/plda_scores

# DNN PLDA EER
# ind pooled: 1.20
# ind female: 1.46
# ind male:   0.87
# dep female: 1.43
# dep male:   0.72
# dep pooled: 1.09

# BNF
# ind female: 1.217
# ind male: 1.097
# ind pooled: 1.138
# dep female: 1.271
# dep male: 0.8949
# dep pooled: 1.145
echo "BNF EER"
for x in ind dep; do
  for y in female male pooled; do
    eer=`compute-eer <(python local/prepare_for_eer.py $trials local/scores_${x}_${y}/plda_scores) 2> /dev/null`
    echo "${x} ${y}: $eer"
  done
done

