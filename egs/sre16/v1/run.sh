#!/bin/bash
# Copyright 2015-2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (EERs) are inline in comments below.

. cmd.sh
. path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
trials=/home/dsnyder/a16/a16/dsnyder/SCALE17/sre16/trials
trials_tgl=/home/dsnyder/a16/a16/dsnyder/SCALE17/sre16/trials_tgl
trials_yue=/home/dsnyder/a16/a16/dsnyder/SCALE17/sre16/trials_yue
num_components=2048
scoring_dir=/home/dsnyder/a16/a16/dsnyder/SCALE17/sre16/sre16_scoring_software/

# Make SRE16 datasets
local/make_sre16_eval.pl /export/corpora5/SRE/R149_0_1 data
local/make_sre16_dev.pl /export/corpora5/SRE/LDC2016E46_SRE16_Call_My_Net_Training_Data data
local/make_sre16_unlabeled.pl /export/corpora5/SRE/LDC2016E46_SRE16_Call_My_Net_Training_Data data

# If you want to use Fisher
# local/make_fisher.sh /export/corpora3/LDC/{LDC2004S13,LDC2004T19} data/fisher1
# local/make_fisher.sh /export/corpora3/LDC/{LDC2005S13,LDC2005T19} data/fisher2
#utils/combine_data.sh data/fisher data/fisher1 data/fisher2

# Prepare a collection of NIST SRE data prior to 2010. This is
# used to train the PLDA model and is also combined with SWB
# for UBM and i-vector extractor training data.
local/make_sre.sh data

# Make SWBD
local/make_swbd_cellular1.pl /export/corpora5/LDC/LDC2001S13 \
  data/swbd_cellular1_train
local/make_swbd_cellular2.pl /export/corpora5/LDC/LDC2004S07 \
  data/swbd_cellular2_train
local/make_swbd2_phase1.pl /export/corpora3/LDC/LDC98S75 \
  data/swbd2_phase1_train
local/make_swbd2_phase2.pl /export/corpora5/LDC/LDC99S79 \
  data/swbd2_phase2_train
local/make_swbd2_phase3.pl /export/corpora5/LDC/LDC2002S06 \
  data/swbd2_phase3_train

utils/combine_data.sh data/swbd \
  data/swbd_cellular1_train data/swbd_cellular2_train \
  data/swbd2_phase1_train data/swbd2_phase2_train data/swbd2_phase3_train


for name in sre swbd sre16_dev_enroll sre16_dev_test sre16_eval_enroll sre16_eval_test sre16_major sre16_minor; do
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    data/${name} exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh data/${name}
  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
    data/${name} exp/make_vad $vaddir
  utils/fix_data_dir.sh data/${name}
done

utils/combine_data.sh data/swbd_sre data/swbd data/sre
utils/combine_data.sh data/sre16_unlabeled data/sre16_major data/sre16_minor

# Train UBM and i-vector extractor.
sid/train_diag_ubm.sh --cmd "$train_cmd --mem 20G" \
  --subsample 1 \
  --nj 40 --num-threads 8 \
  data/sre16_unlabeled $num_components \
  exp/diag_ubm

sid/train_full_ubm.sh --cmd "$train_cmd --mem 25G" \
  --nj 40 --remove-low-count-gaussians false \
  --subsample 1 \
  data/sre16_unlabeled \
  exp/diag_ubm exp/full_ubm

sid/train_ivector_extractor.sh --cmd "$train_cmd --mem 35G" \
  --cleanup false \
  --ivector-dim 600 \
  --num-iters 10 \
  exp/full_ubm/final.ubm data/swbd_sre \
  exp/extractor

sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
  exp/extractor data/sre \
  exp/extractor/ivectors_sre

sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
  exp/extractor data/sre16_major \
  exp/extractor/ivectors_sre16_major

sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
  exp/extractor data/sre16_minor \
  exp/extractor/ivectors_sre16_minor

sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
  exp/extractor data/sre16_dev_enroll \
  exp/extractor/ivectors_sre16_dev_enroll

sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
  exp/extractor data/sre16_dev_test \
  exp/extractor/ivectors_sre16_dev_test

sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
  exp/extractor data/sre16_eval_enroll \
  exp/extractor/ivectors_sre16_eval_enroll

sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
  exp/extractor data/sre16_eval_test \
  exp/extractor/ivectors_sre16_eval_test

$train_cmd exp/extractor/ivectors_sre/log/compute_mean.log \
  ivector-mean scp:exp/extractor/ivectors_sre/ivector.scp \
  exp/extractor/ivectors_sre/mean.vec || exit 1;

$train_cmd exp/extractor/ivectors_sre16_major/log/compute_mean.log \
  ivector-mean scp:exp/extractor/ivectors_sre16_major/ivector.scp \
  exp/extractor/ivectors_sre16_major/mean.vec || exit 1;

# Whitening doesn't seem to help here. But you can compute a whitening
# transform like this:
#  $train_cmd exp/extractor/ivectors_sre16_major/log/compute_whitening.log \
#    est-pca --dim=600 --normalize-variance=true --read-vectors=true \
#    scp:exp/extractor/ivectors_sre16_major/ivector.scp \
#    exp/extractor/ivectors_sre16_major/white.mat || exit 1;

# Train the out-of-domain (OOD) PLDA model on the prior NIST SREs.
$train_cmd exp/extractor/ivectors_sre/log/plda.log \
  ivector-compute-plda ark:data/sre/spk2utt \
  "ark:ivector-subtract-global-mean scp:exp/extractor/ivectors_sre/ivector.scp ark:- | ivector-normalize-length ark:-  ark:- |" \
  exp/extractor/ivectors_sre/plda || exit 1;

# Now we go about discovering the speakers on the unlabelled SRE16 major
# dataset.

# First, we compute the PLDA scores between all i-vectors in  sre16_major.
# We will use this for agglomerative clustering later.  Note that scores
# below --min-score are not written--this is to prevent having to write to
# disk scores that are so low they'll never be included in the same
# cluster anyway.
nj=50
data=data/sre16_major
sdata=data/sre16_major/split$nj
utils/split_data.sh data/sre16_major $nj || exit 1;
$train_cmd --mem 6G JOB=1:$nj exp/extractor/ivectors_sre16_major/log/scores.JOB.log \
  ivector-plda-scoring-dense --min-score=-50 "ivector-copy-plda --smoothing=0.0 exp/extractor/ivectors_sre/plda - |" \
  "ark:utils/filter_scp.pl $sdata/JOB/utt2spk exp/extractor/ivectors_sre16_major/ivector.scp | ivector-subtract-global-mean exp/extractor/ivectors_sre16_major/mean.vec scp:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "ark:utils/filter_scp.pl $data/utt2spk exp/extractor/ivectors_sre16_major/ivector.scp | ivector-subtract-global-mean exp/extractor/ivectors_sre16_major/mean.vec scp:- ark:- | ivector-normalize-length ark:- ark:- |" \
  exp/extractor/ivectors_sre16_major/scores.JOB.txt
cat exp/extractor/ivectors_sre16_major/scores.*.txt > exp/extractor/ivectors_sre16_major/scores.txt
rm exp/extractor/ivectors_sre16_major/scores.*.txt

# Now we attempt to discover the speakers in the unlabelled data.  Here
# we just assume there are 40 speakers.  Alternatively, you could omit
# the option --num-speakers and replace it with --threshold to specify
# a stopping threshold.  The threshold could be estimated by looking
# at the distribution of scores or using the SRE16 dev set.
$train_cmd --mem 6G exp/extractor/ivectors_sre/log/cluster_scores.log \
  agglomerative-cluster --num-speakers=40 exp/extractor/ivectors_sre16_major/scores.txt \
  ark,t:exp/extractor/ivectors_sre16_major/utt2spk || exit 1;

utils/utt2spk_to_spk2utt.pl exp/extractor/ivectors_sre16_major/utt2spk > exp/extractor/ivectors_sre16_major/spk2utt

# Now compute the in-domain (IND) PLDA model, using the discovered
# speaker labels.
$train_cmd exp/extractor/ivectors_sre16_major/log/plda.log \
  ivector-compute-plda ark:exp/extractor/ivectors_sre16_major/spk2utt \
  "ark:ivector-subtract-global-mean scp:exp/extractor/ivectors_sre16_major/ivector.scp ark:- | ivector-normalize-length ark:-  ark:- |" \
  exp/extractor/ivectors_sre16_major/plda || exit 1;

# Now interpolate the models.  Conceptually this is
# factor * PLDA_OOD + (1-factor) * PLDA_IND.  Note that we smooth
# the between-class covariance of PLDA_IND since we only have 40 speakers.
$train_cmd  exp/extractor/log/interpolate_plda.log \
  ivector-interpolate-plda --factor=0.85 exp/extractor/ivectors_sre/plda \
  "ivector-copy-plda --smoothing=1 exp/extractor/ivectors_sre16_major/plda -|" \
  exp/extractor/plda || exit 1;

# Now do PLDA scoring with score normalization.
$train_cmd exp/scores/log/sre16_eval_scoring.log \
  ivector-plda-scoring-snorm --max-comparisons=2272 --normalize-length=true \
  --num-utts=ark:exp/extractor/ivectors_sre16_eval_enroll/num_utts.ark \
  "ivector-copy-plda --smoothing=0.0 exp/extractor/plda - |" \
  "ark:ivector-subtract-global-mean exp/extractor/ivectors_sre16_major/mean.vec scp:exp/extractor/ivectors_sre16_major/ivector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
  "ark:ivector-subtract-global-mean exp/extractor/ivectors_sre16_major/mean.vec scp:exp/extractor/ivectors_sre16_major/ivector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
  "ark:ivector-mean ark:data/sre16_eval_enroll/spk2utt scp:exp/extractor/ivectors_sre16_eval_enroll/ivector.scp ark:- | ivector-subtract-global-mean exp/extractor/ivectors_sre16_major/mean.vec ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "ark:ivector-subtract-global-mean exp/extractor/ivectors_sre16_major/mean.vec scp:exp/extractor/ivectors_sre16_eval_test/ivector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
  "cat '$trials' | cut -d\  --fields=1,2 |" exp/scores/sre16_eval_scores_snorm || exit 1;

echo "modelid    segment    side    LLR" > exp/scores/converted_sre16_eval_scores_snorm
awk '{print $1, $2, "a", $3}' exp/scores/sre16_eval_scores_snorm >> exp/scores/converted_sre16_eval_scores_snorm

# EER 11.12, DCF16 0.653
python3 ${scoring_dir}/scoring.py \
  ${scoring_dir}/eval_config.ini -o exp/scores/converted_sre16_eval_scores_snorm 2>&1 | tee exp/scores/pool_results_snorm

# EER 6.99, DCF16 0.509
python3 ${scoring_dir}/scoring.py \
  ${scoring_dir}/eval_yue_config.ini -o exp/scores/converted_sre16_eval_scores_snorm 2>&1 | tee exp/scores/yue_results_snorm

# EER 15.04, DCF16 0.790
python3 ${scoring_dir}/scoring.py \
  ${scoring_dir}/eval_tgl_config.ini -o exp/scores/converted_sre16_eval_scores_snorm 2>&1 | tee exp/scores/tgl_results_snorm

# Now run the DNN embedding system
local/xvector/run_xvector_xent_1a.sh
