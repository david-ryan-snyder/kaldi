#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#

. cmd.sh
. path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
stage=0
dir=exp/xvector_xent_1b1/

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

# This stage prepares the features.
if [ $stage -le 0 ]; then
  echo "$0: preparing features";
  # The features are 20 dim MFCCs that have been mean-normalized over a
  # 3 second sliding window (like in the iVector system).   We use a modified
  # frame-level VAD to remove silence frames, but only if there are no
  # neighboring speech frames --left-context to the left and --right-context
  # to the right.  This allows the DNN to retain some notion of context.  This
  # is a more efficient alternative to writing spliced features to disk.
  local/xvector/prepare_xvector_feats_cmvn_remove_sil.sh --nj 40 --cmd "$train_cmd" \
    --left-context 2 --right-context 2 \
    data/sre data/sre_no_sil exp/sre_no_sil
  utils/fix_data_dir.sh data/sre_no_sil

  local/xvector/prepare_xvector_feats_cmvn_remove_sil.sh --nj 40 --cmd "$train_cmd" \
    --left-context 2 --right-context 2 \
    data/swbd data/swbd_no_sil exp/swbd_no_sil
  utils/fix_data_dir.sh data/swbd_no_sil

  # Get the number of frames after removing most silence frames.
  local/xvector/get_len.sh --nj 40 --cmd "$train_cmd" \
    data/sre_no_sil exp/sre_no_sil
  local/xvector/get_len.sh --nj 40 --cmd "$train_cmd" \
    data/swbd_no_sil exp/swbd_no_sil

  # Throw out any utterances shorter than 10s (1000 frames).
  # NOTE: for now, it's necessary to keep only utterances that
  # are greater than or equal to the maximum chunk-size in the
  # egs.  In this recipe, the maximum chunk size is just 300 so
  # we could set min_len=300 instead of 1000.
  min_len=500
  mv data/sre_no_sil/utt2len data/sre_no_sil/utt2len.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/sre_no_sil/utt2len.bak > data/sre_no_sil/utt2len

  mv data/swbd_no_sil/utt2len data/swbd_no_sil/utt2len.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/swbd_no_sil/utt2len.bak > data/swbd_no_sil/utt2len

  utils/filter_scp.pl data/sre_no_sil/utt2len data/sre_no_sil/utt2spk > data/sre_no_sil/utt2spk.new
  mv data/sre_no_sil/utt2spk.new data/sre_no_sil/utt2spk
  utils/fix_data_dir.sh data/sre_no_sil

  utils/filter_scp.pl data/swbd_no_sil/utt2len data/swbd_no_sil/utt2spk > data/swbd_no_sil/utt2spk.new
  mv data/swbd_no_sil/utt2spk.new data/swbd_no_sil/utt2spk
  utils/fix_data_dir.sh data/swbd_no_sil

  awk '{print $1, NF-1}' data/sre_no_sil/spk2utt > data/sre_no_sil/spk2num
  awk '{print $1, NF-1}' data/swbd_no_sil/spk2utt > data/swbd_no_sil/spk2num

  # Throw out any SRE speakers with fewer than 4 utterances
  min_num_utts=4
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/sre_no_sil/spk2num | utils/filter_scp.pl - data/sre_no_sil/spk2utt > data/sre_no_sil/spk2utt.new

  # Throw out any SWBD speakers with fewer than 5 utterances (we can afford to be pickier here)
  min_num_utts=5
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/swbd_no_sil/spk2num | utils/filter_scp.pl - data/swbd_no_sil/spk2utt > data/swbd_no_sil/spk2utt.new

  mv data/sre_no_sil/spk2utt.new data/sre_no_sil/spk2utt
  mv data/swbd_no_sil/spk2utt.new data/swbd_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/sre_no_sil/spk2utt > data/sre_no_sil/utt2spk
  utils/spk2utt_to_utt2spk.pl data/swbd_no_sil/spk2utt > data/swbd_no_sil/utt2spk
  utils/fix_data_dir.sh data/sre_no_sil
  utils/fix_data_dir.sh data/swbd_no_sil

  # Combined the swbd and sre data
  utils/combine_data.sh data/swbd_sre_no_sil data/swbd_no_sil data/sre_no_sil
  cat data/sre_no_sil/utt2len data/swbd_no_sil/utt2len > data/swbd_sre_no_sil/utt2len
  utils/filter_scp.pl data/swbd_sre_no_sil/utt2spk data/swbd_sre_no_sil/utt2len > data/swbd_sre_no_sil/utt2len.new
  mv data/swbd_sre_no_sil/utt2len.new data/swbd_sre_no_sil/utt2len
fi

# Create training examples from the features prepared above and
# train the DNN
if [ $stage -le 1 ]; then
  local/xvector/tuning/run_xent_1b1.sh --stage -10 --train-stage -10 \
  --data data/swbd_sre_no_sil --nnet-dir $dir --egs-dir $dir/egs
fi

# Extract embeddings
if [ $stage -le 2 ]; then
  # To extract the embeddings from the network, we create a config which
  # replaces the normal posterior output with an output layer consisting of
  # the two embedding layers, A and B.  Embedding A is an affine layer
  # (tdnn6.affine) immediately on top of a statistics pooling layer.
  # Embedding B is an affine layer (tdnn7.affine) after a nonlinearity.
  #
  echo "output-node name=output input=Append(Offset(tdnn6.affine, 0), Offset(tdnn7.affine, 1))" > $dir/extract.config
  # NOTE: if you only want to extract an embedding from one layer, you
  # could make a config that looks like this
  # echo "output-node name=output input=tdnn6.affine" > $dir/extract.config
  # However, you'll have to modify local/xvector/extract_xent.sh.

  # Extract embeddings for SRE16 dev, enroll, test and the old SREs
  local/xvector/extract_xent.sh --nj 50 --cmd "$train_cmd --mem 4G" \
    --chunk-size 6000 \
    $dir data/sre16_major $dir/xvectors_sre16_major

  local/xvector/extract_xent.sh --nj 50 --cmd "$train_cmd --mem 4G" \
    --chunk-size 6000 \
    $dir data/sre16_eval_enroll $dir/xvectors_sre16_eval_enroll

  local/xvector/extract_xent.sh --nj 50 --cmd "$train_cmd --mem 4G" \
    --chunk-size 6000 \
    $dir data/sre16_eval_test $dir/xvectors_sre16_eval_test

  local/xvector/extract_xent.sh --nj 50 --cmd "$train_cmd --mem 6G" \
    --chunk-size 6000 \
    $dir data/sre $dir/xvectors_sre

  local/xvector/extract_xent.sh --nj 50 --cmd "$train_cmd --mem 4G" \
    --chunk-size 6000 \
    $dir data/sitw_eval_enroll $dir/xvectors_sitw_eval_enroll

  local/xvector/extract_xent.sh --nj 50 --cmd "$train_cmd --mem 4G" \
    --chunk-size 6000 \
    $dir data/sitw_eval_test $dir/xvectors_sitw_eval_test

  local/xvector/extract_xent.sh --nj 50 --cmd "$train_cmd --mem 4G" \
    --chunk-size 6000 \
    $dir data/sitw_dev_enroll $dir/xvectors_sitw_dev_enroll

  local/xvector/extract_xent.sh --nj 50 --cmd "$train_cmd --mem 4G" \
    --chunk-size 6000 \
    $dir data/sitw_dev_test $dir/xvectors_sitw_dev_test
fi

# Train LDA and PLDA models on the DNN
if [ $stage -le 3 ]; then
  echo "$0: computing LDA"
  lda_dim_a=100
  lda_dim_b=75
  interpolate_factor=0.5
  # Compute LDA
  $train_cmd $dir/compute_lda_a.log \
    ivector-compute-lda --total-covariance-factor=1.0 --dim=$lda_dim_a \
    scp:$dir/xvectors_sre/xvector_a.scp \
    ark:data/sre/utt2spk ${dir}/transform_a.mat || exit 1;

  $train_cmd $dir/compute_lda_b.log \
    ivector-compute-lda --total-covariance-factor=1.0 --dim=$lda_dim_b \
    scp:$dir/xvectors_sre/xvector_b.scp \
    ark:data/sre/utt2spk ${dir}/transform_b.mat || exit 1;

  echo "$0: training out-of-domain PLDA models"
  # Train OOD PLDA on SRE
  $train_cmd $dir/log/plda_sre_a.log \
    ivector-compute-plda ark:data/sre/spk2utt \
    "ark:ivector-subtract-global-mean scp:$dir/xvectors_sre/xvector_a.scp  ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    ${dir}/plda_a || exit 1;

  $train_cmd $dir/log/plda_sre_b.log \
    ivector-compute-plda ark:data/sre/spk2utt \
    "ark:ivector-subtract-global-mean scp:$dir/xvectors_sre/xvector_b.scp  ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    ${dir}/plda_b || exit 1;
fi

# Compute PLDA scores
if [ $stage -le 4 ]; then
  # Compute scores for embeddings A
  $train_cmd exp/scores/log/sitw_dev_core_scoring_xvector_a.log \
    ivector-plda-scoring --normalize-length=true \
      --num-utts=ark:exp/extractor/ivectors_sitw_dev_enroll/num_utts.ark \
     "ivector-copy-plda --smoothing=0.0 ${dir}/plda_a - |" \
     "ark:ivector-mean ark:data/sitw_dev_enroll/spk2utt scp:${dir}/xvectors_sitw_dev_enroll/xvector_a.scp ark:- | ivector-subtract-global-mean ${dir}/xvectors_sre/xvector_a_mean.vec ark:- ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_sre/xvector_a_mean.vec scp:${dir}/xvectors_sitw_dev_test/xvector_a.scp ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "cat '$sitw_dev_trials_core' | cut -d\  --fields=1,2 |" exp/scores/sitw_dev_core_scores_xvector_a || exit 1;

  $train_cmd exp/scores/log/sitw_dev_assist_scoring_xvector_a.log \
    ivector-plda-scoring --normalize-length=true \
      --num-utts=ark:exp/extractor/ivectors_sitw_dev_enroll/num_utts.ark \
     "ivector-copy-plda --smoothing=0.0 ${dir}/plda_a - |" \
     "ark:ivector-mean ark:data/sitw_dev_enroll/spk2utt scp:${dir}/xvectors_sitw_dev_enroll/xvector_a.scp ark:- | ivector-subtract-global-mean ${dir}/xvectors_sre/xvector_a_mean.vec ark:- ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_sre/xvector_a_mean.vec scp:${dir}/xvectors_sitw_dev_test/xvector_a.scp ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "cat '$sitw_dev_trials_assist' | cut -d\  --fields=1,2 |" exp/scores/sitw_dev_assist_scores_xvector_a || exit 1;

  $train_cmd exp/scores/log/sitw_dev_core_scoring_xvector_b.log \
    ivector-plda-scoring --normalize-length=true \
      --num-utts=ark:exp/extractor/ivectors_sitw_dev_enroll/num_utts.ark \
     "ivector-copy-plda --smoothing=0.0 ${dir}/plda_b - |" \
     "ark:ivector-mean ark:data/sitw_dev_enroll/spk2utt scp:${dir}/xvectors_sitw_dev_enroll/xvector_b.scp ark:- | ivector-subtract-global-mean ${dir}/xvectors_sre/xvector_b_mean.vec ark:- ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_sre/xvector_b_mean.vec scp:${dir}/xvectors_sitw_dev_test/xvector_b.scp ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "cat '$sitw_dev_trials_core' | cut -d\  --fields=1,2 |" exp/scores/sitw_dev_core_scores_xvector_b || exit 1;

  $train_cmd exp/scores/log/sitw_dev_assist_scoring_xvector_b.log \
    ivector-plda-scoring --normalize-length=true \
      --num-utts=ark:exp/extractor/ivectors_sitw_dev_enroll/num_utts.ark \
     "ivector-copy-plda --smoothing=0.0 ${dir}/plda_b - |" \
     "ark:ivector-mean ark:data/sitw_dev_enroll/spk2utt scp:${dir}/xvectors_sitw_dev_enroll/xvector_b.scp ark:- | ivector-subtract-global-mean ${dir}/xvectors_sre/xvector_b_mean.vec ark:- ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_sre/xvector_b_mean.vec scp:${dir}/xvectors_sitw_dev_test/xvector_b.scp ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "cat '$sitw_dev_trials_assist' | cut -d\  --fields=1,2 |" exp/scores/sitw_dev_assist_scores_xvector_b || exit 1;

  echo "xvector A DEV CORE"
  python ${scoring_software_dir}/scoring.py $sitw_dev_trials_core exp/scores/sitw_dev_core_scores_xvector_a 2>&1 | tee exp/scores/sitw_dev_core_results_xvector_a
  echo "xvector A DEV ASSIST"
  python ${scoring_software_dir}/scoring.py $sitw_dev_trials_assist exp/scores/sitw_dev_assist_scores_xvector_a 2>&1 | tee exp/scores/sitw_dev_assist_results_xvector_a

  # Compute the average of the assist and core DCF (helpful for tuning)
  dev_core_dcf=`grep -oP "(?<=minDCF1e-2: )[^ ]+" exp/scores/sitw_dev_core_results_xvector_a`
  dev_assist_dcf=`grep -oP "(?<=minDCF1e-2: )[^ ]+" exp/scores/sitw_dev_assist_results_xvector_a`
  avg_dcf=`echo "print ($dev_core_dcf + $dev_assist_dcf)/2.0" | python`
  echo "xvector A DEV AVG"
  echo "minDCF1e-2: $avg_dcf"

  # Now lets handle embedding B
  echo "xvector B DEV CORE"
  python ${scoring_software_dir}/scoring.py $sitw_dev_trials_core exp/scores/sitw_dev_core_scores_xvector_b 2>&1 | tee exp/scores/sitw_dev_core_results_xvector_b
  echo "xvector B DEV ASSIST"
  python ${scoring_software_dir}/scoring.py $sitw_dev_trials_assist exp/scores/sitw_dev_assist_scores_xvector_b 2>&1 | tee exp/scores/sitw_dev_assist_results_xvector_b

  # Compute the average of the assist and core DCF. This was used to tune the LDA dim and fusion weights
  dev_core_dcf=`grep -oP "(?<=minDCF1e-2: )[^ ]+" exp/scores/sitw_dev_core_results_xvector_b`
  dev_assist_dcf=`grep -oP "(?<=minDCF1e-2: )[^ ]+" exp/scores/sitw_dev_assist_results_xvector_b`
  avg_dcf=`echo "print ($dev_core_dcf + $dev_assist_dcf)/2.0" | python`
  echo "xvector B DEV AVG"
  echo "minDCF1e-2: $avg_dcf"

  echo "$0: combined xvectors (A+B) DEV CORE"
  python local/combine_scores.py 2 exp/scores/sitw_dev_core_scores_xvector_a exp/scores/sitw_dev_core_scores_xvector_b > exp/scores/sitw_dev_core_scores_xvector
  python ${scoring_software_dir}/scoring.py $sitw_dev_trials_core exp/scores/sitw_dev_core_scores_xvector 2>&1 | tee exp/scores/sitw_dev_core_results_xvector
  echo "$0: combined xvectors (A+B) DEV ASSIST"
  python local/combine_scores.py 2 exp/scores/sitw_dev_assist_scores_xvector_a exp/scores/sitw_dev_assist_scores_xvector_b > exp/scores/sitw_dev_assist_scores_xvector
  python ${scoring_software_dir}/scoring.py $sitw_dev_trials_assist exp/scores/sitw_dev_assist_scores_xvector 2>&1 | tee exp/scores/sitw_dev_assist_results_xvector

  dev_core_dcf=`grep -oP "(?<=minDCF1e-2: )[^ ]+" exp/scores/sitw_dev_core_results_xvector`
  dev_assist_dcf=`grep -oP "(?<=minDCF1e-2: )[^ ]+" exp/scores/sitw_dev_assist_results_xvector`
  avg_dcf=`echo "print ($dev_core_dcf + $dev_assist_dcf)/2.0" | python`
  echo "combined xvector (A+B) DEV AVG"
  echo "minDCF1e-2: $avg_dcf"

  # Compute scores for embeddings A
  $train_cmd exp/scores/log/sre16_eval_core_scoring_xvector_a.log \
    ivector-plda-scoring --normalize-length=true \
      --num-utts=ark:exp/extractor/ivectors_sre16_eval_enroll/num_utts.ark \
     "ivector-copy-plda --smoothing=0.0 ${dir}/plda_a - |" \
     "ark:ivector-mean ark:data/sre16_eval_enroll/spk2utt scp:${dir}/xvectors_sre16_eval_enroll/xvector_a.scp ark:- | ivector-subtract-global-mean ${dir}/xvectors_sre/xvector_a_mean.vec ark:- ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_sre/xvector_a_mean.vec scp:${dir}/xvectors_sre16_eval_test/xvector_a.scp ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "cat '$sre16_trials_yue' | cut -d\  --fields=1,2 |" exp/scores/sre16_eval_scores_xvector_a || exit 1;

  $train_cmd exp/scores/log/sitw_eval_core_scoring_xvector_a.log \
    ivector-plda-scoring --normalize-length=true \
      --num-utts=ark:exp/extractor/ivectors_sitw_eval_enroll/num_utts.ark \
     "ivector-copy-plda --smoothing=0.0 ${dir}/plda_a - |" \
     "ark:ivector-mean ark:data/sitw_eval_enroll/spk2utt scp:${dir}/xvectors_sitw_eval_enroll/xvector_a.scp ark:- | ivector-subtract-global-mean ${dir}/xvectors_sre/xvector_a_mean.vec ark:- ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_sre/xvector_a_mean.vec scp:${dir}/xvectors_sitw_eval_test/xvector_a.scp ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "cat '$sitw_eval_trials_core' | cut -d\  --fields=1,2 |" exp/scores/sitw_eval_core_scores_xvector_a || exit 1;

  $train_cmd exp/scores/log/sitw_eval_assist_scoring_xvector_a.log \
    ivector-plda-scoring --normalize-length=true \
      --num-utts=ark:exp/extractor/ivectors_sitw_eval_enroll/num_utts.ark \
     "ivector-copy-plda --smoothing=0.0 ${dir}/plda_a - |" \
     "ark:ivector-mean ark:data/sitw_eval_enroll/spk2utt scp:${dir}/xvectors_sitw_eval_enroll/xvector_a.scp ark:- | ivector-subtract-global-mean ${dir}/xvectors_sre/xvector_a_mean.vec ark:- ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_sre/xvector_a_mean.vec scp:${dir}/xvectors_sitw_eval_test/xvector_a.scp ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "cat '$sitw_eval_trials_assist' | cut -d\  --fields=1,2 |" exp/scores/sitw_eval_assist_scores_xvector_a || exit 1;

  $train_cmd exp/scores/log/sre16_eval_core_scoring_xvector_b.log \
    ivector-plda-scoring --normalize-length=true \
      --num-utts=ark:exp/extractor/ivectors_sre16_eval_enroll/num_utts.ark \
     "ivector-copy-plda --smoothing=0.0 ${dir}/plda_b - |" \
     "ark:ivector-mean ark:data/sre16_eval_enroll/spk2utt scp:${dir}/xvectors_sre16_eval_enroll/xvector_b.scp ark:- | ivector-subtract-global-mean ${dir}/xvectors_sre/xvector_b_mean.vec ark:- ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_sre/xvector_b_mean.vec scp:${dir}/xvectors_sre16_eval_test/xvector_b.scp ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "cat '$sre16_trials_yue' | cut -d\  --fields=1,2 |" exp/scores/sre16_eval_scores_xvector_b || exit 1;

  $train_cmd exp/scores/log/sitw_eval_core_scoring_xvector_b.log \
    ivector-plda-scoring --normalize-length=true \
      --num-utts=ark:exp/extractor/ivectors_sitw_eval_enroll/num_utts.ark \
     "ivector-copy-plda --smoothing=0.0 ${dir}/plda_b - |" \
     "ark:ivector-mean ark:data/sitw_eval_enroll/spk2utt scp:${dir}/xvectors_sitw_eval_enroll/xvector_b.scp ark:- | ivector-subtract-global-mean ${dir}/xvectors_sre/xvector_b_mean.vec ark:- ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_sre/xvector_b_mean.vec scp:${dir}/xvectors_sitw_eval_test/xvector_b.scp ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "cat '$sitw_eval_trials_core' | cut -d\  --fields=1,2 |" exp/scores/sitw_eval_core_scores_xvector_b || exit 1;

  $train_cmd exp/scores/log/sitw_eval_assist_scoring_xvector_b.log \
    ivector-plda-scoring --normalize-length=true \
      --num-utts=ark:exp/extractor/ivectors_sitw_eval_enroll/num_utts.ark \
     "ivector-copy-plda --smoothing=0.0 ${dir}/plda_b - |" \
     "ark:ivector-mean ark:data/sitw_eval_enroll/spk2utt scp:${dir}/xvectors_sitw_eval_enroll/xvector_b.scp ark:- | ivector-subtract-global-mean ${dir}/xvectors_sre/xvector_b_mean.vec ark:- ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_sre/xvector_b_mean.vec scp:${dir}/xvectors_sitw_eval_test/xvector_b.scp ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "cat '$sitw_eval_trials_assist' | cut -d\  --fields=1,2 |" exp/scores/sitw_eval_assist_scores_xvector_b || exit 1;

  echo "xvector A EVAL SRE16"
  python ${scoring_software_dir}/scoring.py $sre16_trials_yue exp/scores/sre16_eval_scores_xvector_a 2>&1 | tee exp/scores/sre16_eval_results_xvector_a

  echo "xvector A EVAL CORE"
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_xvector_a 2>&1 | tee exp/scores/sitw_eval_core_results_xvector_a
  echo "xvector A EVAL ASSIST"
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_assist exp/scores/sitw_eval_assist_scores_xvector_a 2>&1 | tee exp/scores/sitw_eval_assist_results_xvector_a

  eval_sre16_dcf=`grep -oP "(?<=minDCF1e-2: )[^ ]+" exp/scores/sre16_eval_results_xvector_a`
  eval_core_dcf=`grep -oP "(?<=minDCF1e-2: )[^ ]+" exp/scores/sitw_eval_core_results_xvector_a`
  eval_assist_dcf=`grep -oP "(?<=minDCF1e-2: )[^ ]+" exp/scores/sitw_eval_assist_results_xvector_a`
  avg_dcf=`echo "print ($eval_sre16_dcf + $eval_core_dcf + $eval_assist_dcf)/3.0" | python`
  echo "xvector A EVAL AVG"
  echo "minDCF1e-2: $avg_dcf"

  # Now lets handle embedding B
  echo "xvector B EVAL SRE16"
  python ${scoring_software_dir}/scoring.py $sre16_trials_yue exp/scores/sre16_eval_scores_xvector_b 2>&1 | tee exp/scores/sre16_eval_results_xvector_b
  echo "xvector B EVAL CORE"
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_xvector_b 2>&1 | tee exp/scores/sitw_eval_core_results_xvector_b
  echo "xvector B EVAL ASSIST"
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_assist exp/scores/sitw_eval_assist_scores_xvector_b 2>&1 | tee exp/scores/sitw_eval_assist_results_xvector_b

  eval_sre16_dcf=`grep -oP "(?<=minDCF1e-2: )[^ ]+" exp/scores/sre16_eval_results_xvector_b`
  eval_core_dcf=`grep -oP "(?<=minDCF1e-2: )[^ ]+" exp/scores/sitw_eval_core_results_xvector_b`
  eval_assist_dcf=`grep -oP "(?<=minDCF1e-2: )[^ ]+" exp/scores/sitw_eval_assist_results_xvector_b`
  avg_dcf=`echo "print ($eval_sre16_dcf + $eval_core_dcf + $eval_assist_dcf)/3.0" | python`
  echo "xvector B EVAL AVG"
  echo "minDCF1e-2: $avg_dcf"

  echo "$0: combined xvectors (A+B) EVAL SRE16"
  python local/combine_scores.py 2 exp/scores/sre16_eval_scores_xvector_a exp/scores/sre16_eval_scores_xvector_b > exp/scores/sre16_eval_scores_xvector
  python ${scoring_software_dir}/scoring.py $sre16_trials_yue exp/scores/sre16_eval_scores_xvector 2>&1 | tee exp/scores/sre16_eval_results_xvector

  echo "$0: combined xvectors (A+B) EVAL CORE"
  python local/combine_scores.py 2 exp/scores/sitw_eval_core_scores_xvector_a exp/scores/sitw_eval_core_scores_xvector_b > exp/scores/sitw_eval_core_scores_xvector
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_xvector 2>&1 | tee exp/scores/sitw_eval_core_results_xvector
  echo "$0: combined xvectors (A+B) EVAL ASSIST"
  python local/combine_scores.py 2 exp/scores/sitw_eval_assist_scores_xvector_a exp/scores/sitw_eval_assist_scores_xvector_b > exp/scores/sitw_eval_assist_scores_xvector
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_assist exp/scores/sitw_eval_assist_scores_xvector 2>&1 | tee exp/scores/sitw_eval_assist_results_xvector

  eval_sre16_dcf=`grep -oP "(?<=minDCF1e-2: )[^ ]+" exp/scores/sre16_eval_results_xvector`
  eval_core_dcf=`grep -oP "(?<=minDCF1e-2: )[^ ]+" exp/scores/sitw_eval_core_results_xvector`
  eval_assist_dcf=`grep -oP "(?<=minDCF1e-2: )[^ ]+" exp/scores/sitw_eval_assist_results_xvector`
  avg_dcf=`echo "print ($eval_sre16_dcf + $eval_core_dcf + $eval_assist_dcf)/3.0" | python`
  echo "combined xvector (A+B) EVAL AVG"
  echo "minDCF1e-2: $avg_dcf"

  mkdir -p $dir/xvectors_adapt/
  cat $dir/xvectors_{sitw_dev_test,sitw_dev_enroll,sre16_major}/xvector_a.scp > $dir/xvectors_adapt/xvector_a.scp
  cat $dir/xvectors_{sitw_dev_test,sitw_dev_enroll,sre16_major}/xvector_b.scp > $dir/xvectors_adapt/xvector_b.scp
  ### Compute the mean for this adapt set
  $train_cmd $dir/xvectors_adapt/log/compute_mean_a.log \
    ivector-mean scp:$dir/xvectors_adapt/xvector_a.scp \
    $dir/xvectors_adapt/mean_a.vec || exit 1;
  ### Compute the mean for this adapt set
  $train_cmd $dir/xvectors_adapt/log/compute_mean_b.log \
    ivector-mean scp:$dir/xvectors_adapt/xvector_b.scp \
    $dir/xvectors_adapt/mean_b.vec || exit 1;

  # Compute scores for embeddings A
  $train_cmd exp/scores/log/sre16_eval_core_scoring_xvector_a_snorm.log \
    ivector-plda-scoring-snorm --normalize-length=true --max-comparisons=6711 \
      --num-utts=ark:exp/extractor/ivectors_sre16_eval_enroll/num_utts.ark \
     "ivector-copy-plda --smoothing=0.0 ${dir}/plda_a - |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_a.vec scp:${dir}/xvectors_adapt/xvector_a.scp ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_a.vec scp:${dir}/xvectors_adapt/xvector_a.scp ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "ark:ivector-mean ark:data/sre16_eval_enroll/spk2utt scp:${dir}/xvectors_sre16_eval_enroll/xvector_a.scp ark:- | ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_a.vec ark:- ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_a.vec scp:${dir}/xvectors_sre16_eval_test/xvector_a.scp ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "cat '$sre16_trials_yue' | cut -d\  --fields=1,2 |" exp/scores/sre16_eval_scores_xvector_a_snorm || exit 1;

  echo "xvector A EVAL SRE16 (snorm)"
  python ${scoring_software_dir}/scoring.py $sre16_trials_yue exp/scores/sre16_eval_scores_xvector_a_snorm 2>&1 | tee exp/scores/sre16_eval_results_xvector_a_snorm

  $train_cmd exp/scores/log/sitw_eval_core_scoring_xvector_a_snorm.log \
    ivector-plda-scoring-snorm --normalize-length=true --max-comparisons=6711 \
      --num-utts=ark:exp/extractor/ivectors_sitw_eval_enroll/num_utts.ark \
     "ivector-copy-plda --smoothing=0.0 ${dir}/plda_a - |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_a.vec scp:${dir}/xvectors_adapt/xvector_a.scp ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_a.vec scp:${dir}/xvectors_adapt/xvector_a.scp ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "ark:ivector-mean ark:data/sitw_eval_enroll/spk2utt scp:${dir}/xvectors_sitw_eval_enroll/xvector_a.scp ark:- | ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_a.vec ark:- ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_a.vec scp:${dir}/xvectors_sitw_eval_test/xvector_a.scp ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "cat '$sitw_eval_trials_core' | cut -d\  --fields=1,2 |" exp/scores/sitw_eval_core_scores_xvector_a_snorm || exit 1;

  echo "xvector A EVAL CORE (snorm)"
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_xvector_a_snorm 2>&1 | tee exp/scores/sitw_eval_core_results_xvector_a_snorm

  $train_cmd exp/scores/log/sitw_eval_assist_scoring_xvector_a_snorm.log \
    ivector-plda-scoring-snorm --normalize-length=true --max-comparisons=6711 \
      --num-utts=ark:exp/extractor/ivectors_sitw_eval_enroll/num_utts.ark \
     "ivector-copy-plda --smoothing=0.0 ${dir}/plda_a - |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_a.vec scp:${dir}/xvectors_adapt/xvector_a.scp ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_a.vec scp:${dir}/xvectors_adapt/xvector_a.scp ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "ark:ivector-mean ark:data/sitw_eval_enroll/spk2utt scp:${dir}/xvectors_sitw_eval_enroll/xvector_a.scp ark:- | ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_a.vec ark:- ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_a.vec scp:${dir}/xvectors_sitw_eval_test/xvector_a.scp ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "cat '$sitw_eval_trials_assist' | cut -d\  --fields=1,2 |" exp/scores/sitw_eval_assist_scores_xvector_a_snorm || exit 1;

  echo "xvector A EVAL ASSIST (snorm)"
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_assist exp/scores/sitw_eval_assist_scores_xvector_a_snorm 2>&1 | tee exp/scores/sitw_eval_assist_results_xvector_a_snorm

  # Compute scores for embeddings B
  $train_cmd exp/scores/log/sre16_eval_core_scoring_xvector_b_snorm.log \
    ivector-plda-scoring-snorm --normalize-length=true --max-comparisons=6711 \
      --num-utts=ark:exp/extractor/ivectors_sre16_eval_enroll/num_utts.ark \
     "ivector-copy-plda --smoothing=0.0 ${dir}/plda_b - |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_b.vec scp:${dir}/xvectors_adapt/xvector_b.scp ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_b.vec scp:${dir}/xvectors_adapt/xvector_b.scp ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "ark:ivector-mean ark:data/sre16_eval_enroll/spk2utt scp:${dir}/xvectors_sre16_eval_enroll/xvector_b.scp ark:- | ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_b.vec ark:- ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_b.vec scp:${dir}/xvectors_sre16_eval_test/xvector_b.scp ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "cat '$sre16_trials_yue' | cut -d\  --fields=1,2 |" exp/scores/sre16_eval_scores_xvector_b_snorm || exit 1;

  $train_cmd exp/scores/log/sitw_eval_core_scoring_xvector_b_snorm.log \
    ivector-plda-scoring-snorm --normalize-length=true --max-comparisons=6711 \
      --num-utts=ark:exp/extractor/ivectors_sitw_eval_enroll/num_utts.ark \
     "ivector-copy-plda --smoothing=0.0 ${dir}/plda_b - |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_b.vec scp:${dir}/xvectors_adapt/xvector_b.scp ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_b.vec scp:${dir}/xvectors_adapt/xvector_b.scp ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "ark:ivector-mean ark:data/sitw_eval_enroll/spk2utt scp:${dir}/xvectors_sitw_eval_enroll/xvector_b.scp ark:- | ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_b.vec ark:- ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_b.vec scp:${dir}/xvectors_sitw_eval_test/xvector_b.scp ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "cat '$sitw_eval_trials_core' | cut -d\  --fields=1,2 |" exp/scores/sitw_eval_core_scores_xvector_b_snorm || exit 1;

  $train_cmd exp/scores/log/sitw_eval_assist_scoring_xvector_b_snorm.log \
    ivector-plda-scoring-snorm --normalize-length=true --max-comparisons=6711 \
      --num-utts=ark:exp/extractor/ivectors_sitw_eval_enroll/num_utts.ark \
     "ivector-copy-plda --smoothing=0.0 ${dir}/plda_b - |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_b.vec scp:${dir}/xvectors_adapt/xvector_b.scp ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_b.vec scp:${dir}/xvectors_adapt/xvector_b.scp ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "ark:ivector-mean ark:data/sitw_eval_enroll/spk2utt scp:${dir}/xvectors_sitw_eval_enroll/xvector_b.scp ark:- | ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_b.vec ark:- ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_adapt/mean_b.vec scp:${dir}/xvectors_sitw_eval_test/xvector_b.scp ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "cat '$sitw_eval_trials_assist' | cut -d\  --fields=1,2 |" exp/scores/sitw_eval_assist_scores_xvector_b_snorm || exit 1;

  echo "ivector EVAL SRE16 (with snorm)"
  python ${scoring_software_dir}/scoring.py $sre16_trials_yue exp/scores/sre16_eval_yue_scores_snorm 2>&1 | tee exp/scores/sre16_eval_yue_results_snorm
  echo "ivector EVAL CORE (with snorm)"
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_snorm 2>&1 | tee exp/scores/sitw_eval_core_results_snorm
  echo "ivector EVAL ASSIST (with snorm)"
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_assist exp/scores/sitw_eval_assist_scores_snorm 2>&1 | tee exp/scores/sitw_eval_assist_results_snorm

  echo "xvector A EVAL SRE16 (snorm)"
  python ${scoring_software_dir}/scoring.py $sre16_trials_yue exp/scores/sre16_eval_scores_xvector_a_snorm 2>&1 | tee exp/scores/sre16_eval_results_xvector_a_snorm

  echo "xvector A EVAL CORE (snorm)"
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_xvector_a_snorm 2>&1 | tee exp/scores/sitw_eval_core_results_xvector_a_snorm
  echo "xvector A EVAL ASSIST (snorm)"
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_assist exp/scores/sitw_eval_assist_scores_xvector_a_snorm 2>&1 | tee exp/scores/sitw_eval_assist_results_xvector_a_snorm

  echo "xvector B EVAL SRE16 (snorm)"
  python ${scoring_software_dir}/scoring.py $sre16_trials_yue exp/scores/sre16_eval_scores_xvector_b_snorm 2>&1 | tee exp/scores/sre16_eval_results_xvector_b_snorm
  echo "xvector B EVAL CORE (snorm)"
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_xvector_b_snorm 2>&1 | tee exp/scores/sitw_eval_core_results_xvector_b_snorm
  echo "xvector B EVAL ASSIST (snorm)"
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_assist exp/scores/sitw_eval_assist_scores_xvector_b_snorm 2>&1 | tee exp/scores/sitw_eval_assist_results_xvector_b_snorm

  echo "$0: combined xvectors (A+B) EVAL SRE16 (snorm)"
  python local/combine_scores.py 2 exp/scores/sre16_eval_scores_xvector_a_snorm exp/scores/sre16_eval_scores_xvector_b_snorm > exp/scores/sre16_eval_scores_xvector_snorm
  python ${scoring_software_dir}/scoring.py $sre16_trials_yue exp/scores/sre16_eval_scores_xvector_snorm 2>&1 | tee exp/scores/sre16_eval_results_xvector_snorm
  echo "$0: combined xvectors (A+B) EVAL CORE (snorm)"
  python local/combine_scores.py 2 exp/scores/sitw_eval_core_scores_xvector_a_snorm exp/scores/sitw_eval_core_scores_xvector_b_snorm > exp/scores/sitw_eval_core_scores_xvector_snorm
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_xvector_snorm 2>&1 | tee exp/scores/sitw_eval_core_results_xvector_snorm
  echo "$0: combined xvectors (A+B) EVAL ASSIST (snorm)"
  python local/combine_scores.py 2 exp/scores/sitw_eval_assist_scores_xvector_a_snorm exp/scores/sitw_eval_assist_scores_xvector_b_snorm > exp/scores/sitw_eval_assist_scores_xvector_snorm
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_assist exp/scores/sitw_eval_assist_scores_xvector_snorm 2>&1 | tee exp/scores/sitw_eval_assist_results_xvector_snorm

  eval_sre16_dcf=`grep -oP "(?<=minDCF1e-2: )[^ ]+" exp/scores/sre16_eval_results_xvector_snorm`
  eval_core_dcf=`grep -oP "(?<=minDCF1e-2: )[^ ]+" exp/scores/sitw_eval_core_results_xvector_snorm`
  eval_assist_dcf=`grep -oP "(?<=minDCF1e-2: )[^ ]+" exp/scores/sitw_eval_assist_results_xvector_snorm`
  avg_dcf=`echo "print ($eval_sre16_dcf + $eval_core_dcf + $eval_assist_dcf)/3.0" | python`
  echo "combined xvector (A+B) EVAL AVG (snorm)"
  echo "minDCF1e-2: $avg_dcf"

  sitw_eval_trials_core=data/sitw_eval_test/trials/aux/core-core.lt15s-tst.lst
  echo
  echo "ivector (snorm) lt15ss"
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_snorm 2>&1 | tee exp/scores/sitw_eval_core_results_snorm
  echo "xvector A EVAL CORE (snorm) lt15s"
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_xvector_a_snorm 2>&1 | tee exp/scores/sitw_eval_core_results_xvector_a_snorm
  echo "xvector B EVAL CORE (snorm) lt15s"
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_xvector_b_snorm 2>&1 | tee exp/scores/sitw_eval_core_results_xvector_b_snorm
  echo "$0: combined xvectors (A+B) EVAL CORE (snorm) lt15s"
  python local/combine_scores.py 2 exp/scores/sitw_eval_core_scores_xvector_a_snorm exp/scores/sitw_eval_core_scores_xvector_b_snorm > exp/scores/sitw_eval_core_scores_xvector_snorm
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_xvector_snorm 2>&1 | tee exp/scores/sitw_eval_core_results_xvector_snorm
  echo "$0 combing with ivectors lt15s"
  python local/combine_scores.py 2 exp/scores/sitw_eval_core_scores_xvector_snorm exp/scores/sitw_eval_core_scores_snorm > exp/scores/sitw_eval_core_scores_fusion_snorm
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_fusion_snorm 2>&1 | tee exp/scores/sitw_eval_core_results_fusion_snorm

  sitw_eval_trials_core=data/sitw_eval_test/trials/aux/core-core.15to25s-tst.lst
  echo
  echo "ivector (snorm) 15-25ss"
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_snorm 2>&1 | tee exp/scores/sitw_eval_core_results_snorm
  echo "xvector A EVAL CORE (snorm) 15-25s"
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_xvector_a_snorm 2>&1 | tee exp/scores/sitw_eval_core_results_xvector_a_snorm
  echo "xvector B EVAL CORE (snorm) 15-25s"
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_xvector_b_snorm 2>&1 | tee exp/scores/sitw_eval_core_results_xvector_b_snorm
  echo "$0: combined xvectors (A+B) EVAL CORE (snorm) 15-25s"
  python local/combine_scores.py 2 exp/scores/sitw_eval_core_scores_xvector_a_snorm exp/scores/sitw_eval_core_scores_xvector_b_snorm > exp/scores/sitw_eval_core_scores_xvector_snorm
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_xvector_snorm 2>&1 | tee exp/scores/sitw_eval_core_results_xvector_snorm
  echo "$0 combing with ivectors 15-25s"
  python local/combine_scores.py 1 exp/scores/sitw_eval_core_scores_xvector_snorm exp/scores/sitw_eval_core_scores_snorm > exp/scores/sitw_eval_core_scores_fusion_snorm
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_fusion_snorm 2>&1 | tee exp/scores/sitw_eval_core_results_fusion_snorm

  sitw_eval_trials_core=data/sitw_eval_test/trials/aux/core-core.25to40s-tst.lst
  echo
  echo "ivector (snorm) 25-40s"
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_snorm 2>&1 | tee exp/scores/sitw_eval_core_results_snorm
  echo "xvector A EVAL CORE (snorm) 25-40s"
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_xvector_a_snorm 2>&1 | tee exp/scores/sitw_eval_core_results_xvector_a_snorm
  echo "xvector B EVAL CORE (snorm) 25-40s"
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_xvector_b_snorm 2>&1 | tee exp/scores/sitw_eval_core_results_xvector_b_snorm
  echo "$0: combined xvectors (A+B) EVAL CORE (snorm) 25-40s"
  python local/combine_scores.py 2 exp/scores/sitw_eval_core_scores_xvector_a_snorm exp/scores/sitw_eval_core_scores_xvector_b_snorm > exp/scores/sitw_eval_core_scores_xvector_snorm
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_xvector_snorm 2>&1 | tee exp/scores/sitw_eval_core_results_xvector_snorm
  echo "$0 combing with ivectors 25-40s"
  python local/combine_scores.py 1 exp/scores/sitw_eval_core_scores_xvector_snorm exp/scores/sitw_eval_core_scores_snorm > exp/scores/sitw_eval_core_scores_fusion_snorm
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_fusion_snorm 2>&1 | tee exp/scores/sitw_eval_core_results_fusion_snorm

  # FUSION WITH IVECTORS
  echo "CORE fusion with ivectors"
  python local/combine_scores.py 1 exp/scores/sitw_eval_core_scores_xvector_snorm exp/scores/sitw_eval_core_scores_snorm > exp/scores/sitw_eval_core_scores_fusion_snorm
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_core exp/scores/sitw_eval_core_scores_fusion_snorm 2>&1 | tee exp/scores/sitw_eval_core_results_fusion_snorm
  echo "ASSIST fusion with ivectors"
  python local/combine_scores.py 1 exp/scores/sitw_eval_assist_scores_xvector_snorm exp/scores/sitw_eval_assist_scores_snorm > exp/scores/sitw_eval_assist_scores_fusion_snorm
  python ${scoring_software_dir}/scoring.py $sitw_eval_trials_assist exp/scores/sitw_eval_assist_scores_fusion_snorm 2>&1 | tee exp/scores/sitw_eval_assist_results_fusion_snorm
  echo "SRE16 fusion with ivectors"
  python local/combine_scores.py 1 exp/scores/sre16_eval_scores_xvector_snorm exp/scores/sre16_eval_yue_scores_snorm > exp/scores/sre16_eval_scores_fusion_snorm
  python ${scoring_software_dir}/scoring.py $sre16_trials_yue exp/scores/sre16_eval_scores_fusion_snorm 2>&1 | tee exp/scores/sre16_eval_results_fusion_snorm

fi
