#!/bin/bash
#
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (EERs) are inline in comments below.

. cmd.sh
. path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

# SITW Trials
sitw_dev_trials_core=data/sitw_dev_test/trials/core-core.lst
sitw_eval_trials_core=data/sitw_eval_test/trials/core-core.lst
sitw_dev_trials_assist=data/sitw_dev_test/trials/assist-multi.lst
sitw_eval_trials_assist=data/sitw_eval_test/trials/assist-multi.lst

# SRE16 trials
sre16_trials=/home/dsnyder/a16/a16/dsnyder/SCALE17/sre16/trials
sre16_trials_tgl=/home/dsnyder/a16/a16/dsnyder/SCALE17/sre16/trials_tgl
sre16_trials_yue=/home/dsnyder/a16/a16/dsnyder/SCALE17/sre16/trials_yue
scoring_software_dir=/home/dsnyder/a16/a16/dsnyder/SCALE17/sitw/scoring_software/

# Make Mixer6. The script prepares all of the telephone calls and microphone
# recordings of the telephone calls using microphone 02.  In addition, we add
# in a random subset of all of the other microphones as well.  The interviews
# are not included here.
local/make_mx6.sh

# Prepare SRE10 test and enroll. Includes microphone interview speech.
local/make_sre10.pl /export/corpora5/SRE/SRE2010/eval/ data/

# Prepare SRE08 test and enroll. Includes microphone speech.
local/make_sre08.pl /export/corpora5/LDC/LDC2011S08 /export/corpora5/LDC/LDC2011S05 data/

# TODO: for now, this prepares just the telephone portion of SRE04-06.
# We'll make a new recipe in ../v2 which uses more microphone speech from
# the SREs.
local/make_sre.sh data/

# Combine all SREs and Mixer6 into one dataset
utils/combine_data.sh data/sre \
  data/sre2004 data/sre2005_train \
  data/sre2005_test data/sre2006_train \
  data/sre2006_test_1 data/sre2006_test_2 \
  data/sre08 data/mx6 data/sre10
utils/validate_data_dir.sh --no-text --no-feats data/sre
utils/fix_data_dir.sh data/sre

# Make SRE16 datasets
local/make_sre16_eval.pl /export/corpora5/SRE/R149_0_1 data
local/make_sre16_dev.pl /export/corpora5/SRE/LDC2016E46_SRE16_Call_My_Net_Training_Data data
local/make_sre16_unlabeled.pl /export/corpora5/SRE/LDC2016E46_SRE16_Call_My_Net_Training_Data data

# Make SITW dev and eval sets
local/make_sitw.sh /export/a15/gsell/sitw data/sitw

# If you want to use Fisher
# local/make_fisher.sh /export/corpora3/LDC/{LDC2004S13,LDC2004T19} data/fisher1
# local/make_fisher.sh /export/corpora3/LDC/{LDC2005S13,LDC2005T19} data/fisher2
# utils/combine_data.sh data/fisher data/fisher1 data/fisher2

# If you want to use SWBD
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

# Make MFCCs and compute the energy-based VAD for each dataset
for name in sre swbd sitw_dev_enroll sitw_dev_test sitw_eval_enroll sitw_eval_test sre16_dev_enroll sre16_dev_test sre16_eval_enroll sre16_eval_test sre16_major sre16_minor; do
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    data/${name} exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh data/${name}
  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
    data/${name} exp/make_vad $vaddir
  utils/fix_data_dir.sh data/${name}
done

utils/combine_data.sh data/swbd_sre data/swbd data/sre
utils/combine_data.sh data/sre16_unlabeled data/sre16_major data/sre16_minor
utils/subset_data_dir.sh data/swbd_sre 16000 data/swbd_sre_16k
utils/subset_data_dir.sh data/swbd_sre 32000 data/swbd_sre_32k

# Train UBM and i-vector extractor.
sid/train_diag_ubm.sh --cmd "$train_cmd --mem 20G" \
  --nj 40 --num-threads 8 \
  data/swbd_sre_16k 2048 \
  exp/diag_ubm

sid/train_full_ubm.sh --cmd "$train_cmd --mem 25G" \
  --nj 40 --remove-low-count-gaussians false \
  data/swbd_sre_32k \
  exp/diag_ubm exp/full_ubm

# Train the i-vector extractor.  Here we're using num-iters=10.
# It's possible you can decrease this to 5 and it may not increase
# error-rate much (and it will finish much faster).
sid/train_ivector_extractor.sh --cmd "$train_cmd --mem 35G" \
  --ivector-dim 600 \
  --num-iters 10 \
  exp/full_ubm/final.ubm data/swbd_sre \
  exp/extractor

# Extract i-vectors for SRE data (includes Mixer 6). We'll use this for things
# like LDA, PLDA and centering.
sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
  exp/extractor data/sre \
  exp/extractor/ivectors_sre

# Extract i-vectors for SRE16 datasets.  This script isn't currently using
# the unlabelled or dev data, but it doesn't cost much to extract it here.
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

# Extract i-vectors for SITW datasets
sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
  exp/extractor data/sitw_dev_test \
  exp/extractor/ivectors_sitw_dev_test

sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
  exp/extractor data/sitw_dev_enroll \
  exp/extractor/ivectors_sitw_dev_enroll

sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
  exp/extractor data/sitw_eval_test \
  exp/extractor/ivectors_sitw_eval_test

sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
  exp/extractor data/sitw_eval_enroll \
  exp/extractor/ivectors_sitw_eval_enroll

$train_cmd exp/extractor/ivectors_sre/log/compute_mean.log \
  ivector-mean scp:exp/extractor/ivectors_sre/ivector.scp \
  exp/extractor/ivectors_sre/mean.vec || exit 1;

# This script uses LDA to decrease the dimensionality prior to PLDA.
# Alternatively, you can use the --dim argument in ivector-copy-plda
# to reduce the dimensionality in the PLDA space.
lda_dim=200
$train_cmd exp/extractor/ivectors_sre/log/lda.log \
  ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
  "ark:ivector-subtract-global-mean scp:exp/extractor/ivectors_sre/ivector.scp ark:- |" \
  ark:data/sre/utt2spk exp/extractor/ivectors_sre/transform.mat || exit 1;

$train_cmd exp/extractor/ivectors_sre/log/plda.log \
  ivector-compute-plda ark:data/sre/spk2utt \
  "ark:ivector-subtract-global-mean scp:exp/extractor/ivectors_sre/ivector.scp ark:- | transform-vec exp/extractor/ivectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
  exp/extractor/ivectors_sre/plda || exit 1;

# SITW dev core-core
$train_cmd exp/scores/log/sitw_dev_core_scoring.log \
  ivector-plda-scoring --normalize-length=true \
  --num-utts=ark:exp/extractor/ivectors_sitw_dev_enroll/num_utts.ark \
  "ivector-copy-plda --smoothing=0.0 exp/extractor/ivectors_sre/plda - |" \
  "ark:ivector-mean ark:data/sitw_dev_enroll/spk2utt scp:exp/extractor/ivectors_sitw_dev_enroll/ivector.scp ark:- | ivector-subtract-global-mean exp/extractor/ivectors_sre/mean.vec ark:- ark:- | transform-vec exp/extractor/ivectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "ark:ivector-subtract-global-mean exp/extractor/ivectors_sre/mean.vec scp:exp/extractor/ivectors_sitw_dev_test/ivector.scp ark:- | transform-vec exp/extractor/ivectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "cat '$sitw_dev_trials_core' | cut -d\  --fields=1,2 |" exp/scores/sitw_dev_core_scores || exit 1;

# SITW dev assist-multi
$train_cmd exp/scores/log/sitw_dev_assist_scoring.log \
  ivector-plda-scoring --normalize-length=true \
  --num-utts=ark:exp/extractor/ivectors_sitw_dev_enroll/num_utts.ark \
  "ivector-copy-plda --smoothing=0.0 exp/extractor/ivectors_sre/plda - |" \
  "ark:ivector-mean ark:data/sitw_dev_enroll/spk2utt scp:exp/extractor/ivectors_sitw_dev_enroll/ivector.scp ark:- | ivector-subtract-global-mean exp/extractor/ivectors_sre/mean.vec ark:- ark:- | transform-vec exp/extractor/ivectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "ark:ivector-subtract-global-mean exp/extractor/ivectors_sre/mean.vec scp:exp/extractor/ivectors_sitw_dev_test/ivector.scp ark:- | transform-vec exp/extractor/ivectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "cat '$sitw_dev_trials_assist' | cut -d\  --fields=1,2 |" exp/scores/sitw_dev_assist_scores || exit 1;

echo "DEV CORE"
python ${scoring_software_dir}/scoring.py $sitw_dev_trials_core exp/scores/sitw_dev_core_scores 2>&1 | tee exp/scores/sitw_dev_core_results
echo "DEV ASSIST"
python ${scoring_software_dir}/scoring.py $sitw_dev_trials_assist exp/scores/sitw_dev_assist_scores 2>&1 | tee exp/scores/sitw_dev_assist_results

# Compute the average of the assist and core DCF (helpful for tuning)
dev_core_dcf=`grep -oP "(?<=minDCF1e-2: )[^ ]+" exp/scores/sitw_dev_core_results`
dev_assist_dcf=`grep -oP "(?<=minDCF1e-2: )[^ ]+" exp/scores/sitw_dev_assist_results`
avg_dcf=`echo "print ($dev_core_dcf + $dev_assist_dcf)/2.0" | python`
echo "DEV AVG"
echo "minDCF1e-2: $avg_dcf"

# Now do PLDA scoring on the Cantonese portion of SRE16
$train_cmd exp/scores/log/sre16_eval_yue_scoring.log \
  ivector-plda-scoring --normalize-length=true \
  --num-utts=ark:exp/extractor/ivectors_sre16_eval_enroll/num_utts.ark \
  "ivector-copy-plda --smoothing=0.0 exp/extractor/ivectors_sre/plda - |" \
  "ark:ivector-mean ark:data/sre16_eval_enroll/spk2utt scp:exp/extractor/ivectors_sre16_eval_enroll/ivector.scp ark:- | ivector-subtract-global-mean exp/extractor/ivectors_sre/mean.vec ark:- ark:- | transform-vec exp/extractor/ivectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "ark:ivector-subtract-global-mean exp/extractor/ivectors_sre/mean.vec scp:exp/extractor/ivectors_sre16_eval_test/ivector.scp ark:- | transform-vec exp/extractor/ivectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "cat '$sre16_trials_yue' | cut -d\  --fields=1,2 |" exp/scores/sre16_eval_yue_scores || exit 1;

# SITW eval core-core
$train_cmd exp/scores/log/sitw_eval_core_scoring.log \
  ivector-plda-scoring --normalize-length=true \
  --num-utts=ark:exp/extractor/ivectors_sitw_eval_enroll/num_utts.ark \
  "ivector-copy-plda --smoothing=0.0 exp/extractor/ivectors_sre/plda - |" \
  "ark:ivector-mean ark:data/sitw_eval_enroll/spk2utt scp:exp/extractor/ivectors_sitw_eval_enroll/ivector.scp ark:- | ivector-subtract-global-mean exp/extractor/ivectors_sre/mean.vec ark:- ark:- | transform-vec exp/extractor/ivectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "ark:ivector-subtract-global-mean exp/extractor/ivectors_sre/mean.vec scp:exp/extractor/ivectors_sitw_eval_test/ivector.scp ark:- | transform-vec exp/extractor/ivectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "cat '$sitw_eval_trials_core' | cut -d\  --fields=1,2 |" exp/scores/sitw_eval_core_scores || exit 1;

# SITW eval assist-multi
$train_cmd exp/scores/log/sitw_eval_assist_scoring.log \
  ivector-plda-scoring --normalize-length=true \
  --num-utts=ark:exp/extractor/ivectors_sitw_eval_enroll/num_utts.ark \
  "ivector-copy-plda --smoothing=0.0 exp/extractor/ivectors_sre/plda - |" \
  "ark:ivector-mean ark:data/sitw_eval_enroll/spk2utt scp:exp/extractor/ivectors_sitw_eval_enroll/ivector.scp ark:- | ivector-subtract-global-mean exp/extractor/ivectors_sre/mean.vec ark:- ark:- | transform-vec exp/extractor/ivectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "ark:ivector-subtract-global-mean exp/extractor/ivectors_sre/mean.vec scp:exp/extractor/ivectors_sitw_eval_test/ivector.scp ark:- | transform-vec exp/extractor/ivectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "cat '$sitw_eval_trials_assist' | cut -d\  --fields=1,2 |" exp/scores/sitw_eval_assist_scores || exit 1;

# SRE16
# EER: 10.85 minDCF1e-3: 0.8449 minDCF1e-2: 0.6829
# CORE
# EER: 10.66 minDCF1e-3: 0.8238 minDCF1e-2: 0.6725
# ASSIST
# EER: 12.82 minDCF1e-3: 0.8388 minDCF1e-2: 0.6998
echo "EVAL SRE16"
python ${scoring_software_dir}/scoring.py $sre16_trials_yue exp/scores/sre16_eval_yue_scores 2>&1 | tee exp/scores/sre16_eval_yue_results
echo "EVAL CORE"
python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores 2>&1 | tee exp/scores/sitw_eval_core_results
echo "EVAL ASSIST"
python ${scoring_software_dir}/scoring.py $sitw_eval_trials_assist exp/scores/sitw_eval_assist_scores 2>&1 | tee exp/scores/sitw_eval_assist_results

# EVAL AVG
# minDCF1e-2: 0.6851
sre16_dcf=`grep -oP "(?<=minDCF1e-2: )[^ ]+" exp/scores/sre16_eval_yue_results`
eval_core_dcf=`grep -oP "(?<=minDCF1e-2: )[^ ]+" exp/scores/sitw_eval_core_results`
eval_assist_dcf=`grep -oP "(?<=minDCF1e-2: )[^ ]+" exp/scores/sitw_eval_assist_results`
avg_dcf=`echo "print ($sre16_dcf + $eval_core_dcf + $eval_assist_dcf)/3.0" | python`
echo "EVAL AVG"
echo "minDCF1e-2: $avg_dcf"
