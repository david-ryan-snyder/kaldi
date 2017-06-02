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
dir=exp/xvector_xent_1a/

trials=/home/dsnyder/a16/a16/dsnyder/SCALE17/sre16/trials
trials_tgl=/home/dsnyder/a16/a16/dsnyder/SCALE17/sre16/trials_tgl
trials_yue=/home/dsnyder/a16/a16/dsnyder/SCALE17/sre16/trials_yue
scoring_dir=/home/dsnyder/a16/a16/dsnyder/SCALE17/sre16/sre16_scoring_software/

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
  min_len=1000
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
  local/xvector/tuning/run_xent_1a.sh --stage -10 --train-stage -1 \
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
  local/xvector/extract_xent.sh --nj 100 --cmd "$train_cmd --mem 4G" \
    $dir data/sre16_major $dir/xvectors_sre16_major

  local/xvector/extract_xent.sh --nj 100 --cmd "$train_cmd --mem 4G" \
    $dir data/sre16_eval_enroll $dir/xvectors_sre16_eval_enroll

  local/xvector/extract_xent.sh --nj 100 --cmd "$train_cmd --mem 4G" \
    $dir data/sre16_eval_test $dir/xvectors_sre16_eval_test

  local/xvector/extract_xent.sh --nj 100 --cmd "$train_cmd --mem 6G" \
    $dir data/sre $dir/xvectors_sre
fi

# Train LDA and PLDA models on the DNN
if [ $stage -le 3 ]; then
  echo "$0: computing LDA"
  lda_dim_a=128
  lda_dim_b=75
  interpolate_factor=0.5
  # Compute LDA
  $train_cmd $dir/compute_lda_a.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim_a \
    scp:$dir/xvectors_sre/xvector_a.scp \
    ark:data/sre/utt2spk ${dir}/transform_a.mat || exit 1;

  $train_cmd $dir/compute_lda_b.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim_b \
    scp:$dir/xvectors_sre/xvector_b.scp \
    ark:data/sre/utt2spk ${dir}/transform_b.mat || exit 1;

  echo "$0: training out-of-domain PLDA models"
  # Train OOD PLDA on SRE
  $train_cmd $dir/log/plda_sre_a.log \
    ivector-compute-plda ark:data/sre/spk2utt \
    "ark:ivector-subtract-global-mean scp:$dir/xvectors_sre/xvector_a.scp  ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    ${dir}/plda_sre_a || exit 1;

  $train_cmd $dir/log/plda_sre_b.log \
    ivector-compute-plda ark:data/sre/spk2utt \
    "ark:ivector-subtract-global-mean scp:$dir/xvectors_sre/xvector_b.scp  ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    ${dir}/plda_sre_b || exit 1;

  echo "$0: training in-domain PLDA models"
  # Train IND PLDA on SRE16 Major
  $train_cmd $dir/log/plda_sre16_major_a.log \
    ivector-compute-plda ark:exp/extractor/ivectors_sre16_major/spk2utt \
    "ark:ivector-subtract-global-mean scp:$dir/xvectors_sre16_major/xvector_a.scp ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ${dir}/plda_sre16_major_a || exit 1;

  $train_cmd $dir/log/plda_sre16_major_b.log \
    ivector-compute-plda ark:exp/extractor/ivectors_sre16_major/spk2utt \
    "ark:ivector-subtract-global-mean scp:$dir/xvectors_sre16_major/xvector_b.scp ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ${dir}/plda_sre16_major_b || exit 1;

  echo "$0: interpolating out-of-domain and in-domain PLDA models"
  # Interpolate the parameters of the OOD PLDA and IND PLDA
  $train_cmd ${dir}/log/interpolate_plda_a.log \
    ivector-interpolate-plda --factor=$interpolate_factor ${dir}/plda_sre_a \
    "ivector-copy-plda --smoothing=1 ${dir}/plda_sre16_major_a -|" \
    ${dir}/plda_a || exit 1;

  $train_cmd ${dir}/log/interpolate_plda_b.log \
    ivector-interpolate-plda --factor=$interpolate_factor ${dir}/plda_sre_b \
    "ivector-copy-plda --smoothing=1 ${dir}/plda_sre16_major_b -|" \
    ${dir}/plda_b || exit 1;
fi

# Compute PLDA scores
if [ $stage -le 4 ]; then
  # Compute scores for embeddings A
  $train_cmd exp/scores/log/sre16_eval_scoring_xvector_a.log \
    ivector-plda-scoring-snorm --max-comparisons=2272 --normalize-length=true \
      --num-utts=ark:exp/extractor/ivectors_sre16_eval_enroll/num_utts.ark \
     "ivector-copy-plda --smoothing=0.0 ${dir}/plda_a - |" \
     "ark:ivector-subtract-global-mean scp:${dir}/xvectors_sre16_major/xvector_a.scp ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
     "ark:ivector-subtract-global-mean scp:${dir}/xvectors_sre16_major/xvector_a.scp ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
     "ark:ivector-mean ark:data/sre16_eval_enroll/spk2utt scp:${dir}/xvectors_sre16_eval_enroll/xvector_a.scp ark:- | ivector-subtract-global-mean ${dir}/xvectors_sre16_major/xvector_a_mean.vec ark:- ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_sre16_major/xvector_a_mean.vec scp:${dir}/xvectors_sre16_eval_test/xvector_a.scp ark:- | transform-vec ${dir}/transform_a.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "cat '$trials' | cut -d\  --fields=1,2 |" exp/scores/sre16_eval_scores_xvector_a_snorm || exit 1;

echo "$0: results for embedding A"
echo "modelid    segment    side    LLR" > exp/scores/converted_sre16_eval_scores_xvector_a_snorm
awk '{print $1, $2, "a", $3}' exp/scores/sre16_eval_scores_xvector_a_snorm >> exp/scores/converted_sre16_eval_scores_xvector_a_snorm
python3 ${scoring_dir}/scoring.py \
  ${scoring_dir}/eval_config.ini -o exp/scores/converted_sre16_eval_scores_xvector_a_snorm 2>&1 | tee exp/scores/pool_results_xvector_a_snorm
python3 ${scoring_dir}/scoring.py \
  ${scoring_dir}/eval_yue_config.ini -o exp/scores/converted_sre16_eval_scores_xvector_a_snorm 2>&1 | tee exp/scores/yue_results_xvector_a_snorm
python3 ${scoring_dir}/scoring.py \
  ${scoring_dir}/eval_tgl_config.ini -o exp/scores/converted_sre16_eval_scores_xvector_a_snorm 2>&1 | tee exp/scores/tgl_results_xvector_a_snorm

# Compute scores for embeddings B
  $train_cmd exp/scores/log/sre16_eval_scoring_xvector_b.log \
    ivector-plda-scoring-snorm --max-comparisons=2272 --normalize-length=true \
      --num-utts=ark:exp/extractor/ivectors_sre16_eval_enroll/num_utts.ark \
     "ivector-copy-plda --smoothing=0.0 ${dir}/plda_b - |" \
     "ark:ivector-subtract-global-mean scp:${dir}/xvectors_sre16_major/xvector_b.scp ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
     "ark:ivector-subtract-global-mean scp:${dir}/xvectors_sre16_major/xvector_b.scp ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
     "ark:ivector-mean ark:data/sre16_eval_enroll/spk2utt scp:${dir}/xvectors_sre16_eval_enroll/xvector_b.scp ark:- | ivector-subtract-global-mean ${dir}/xvectors_sre16_major/xvector_b_mean.vec ark:- ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
     "ark:ivector-subtract-global-mean ${dir}/xvectors_sre16_major/xvector_b_mean.vec scp:${dir}/xvectors_sre16_eval_test/xvector_b.scp ark:- | transform-vec ${dir}/transform_b.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
     "cat '$trials' | cut -d\  --fields=1,2 |" exp/scores/sre16_eval_scores_xvector_b_snorm || exit 1;

echo "$0: results for embedding B"
echo "modelid    segment    side    LLR" > exp/scores/converted_sre16_eval_scores_xvector_b_snorm
awk '{print $1, $2, "a", $3}' exp/scores/sre16_eval_scores_xvector_b_snorm >> exp/scores/converted_sre16_eval_scores_xvector_b_snorm
python3 ${scoring_dir}/scoring.py \
  ${scoring_dir}/eval_config.ini -o exp/scores/converted_sre16_eval_scores_xvector_b_snorm 2>&1 | tee exp/scores/pool_results_xvector_b_snorm
python3 ${scoring_dir}/scoring.py \
  ${scoring_dir}/eval_yue_config.ini -o exp/scores/converted_sre16_eval_scores_xvector_b_snorm 2>&1 | tee exp/scores/yue_results_xvector_b_snorm
python3 ${scoring_dir}/scoring.py \
  ${scoring_dir}/eval_tgl_config.ini -o exp/scores/converted_sre16_eval_scores_xvector_b_snorm 2>&1 | tee exp/scores/tgl_results_xvector_b_snorm

# Combine the scores from embeddings A and B.  Recall that both
# embeddings come from the same DNN, so if we compute one we get
# the other for "free."
echo "$0: results for combined embeddings (A+B)"
python local/combine_scores.py 1.0 exp/scores/sre16_eval_scores_xvector_a_snorm exp/scores/sre16_eval_scores_xvector_b_snorm > exp/scores/sre16_eval_scores_xvector_snorm
echo "modelid    segment    side    LLR" > exp/scores/converted_sre16_eval_scores_xvector_snorm
awk '{print $1, $2, "a", $3}' exp/scores/sre16_eval_scores_xvector_snorm >> exp/scores/converted_sre16_eval_scores_xvector_snorm

# EER 10.45, DCF16 0.627
python3 ${scoring_dir}/scoring.py \
  ${scoring_dir}/eval_config.ini -o exp/scores/converted_sre16_eval_scores_xvector_snorm 2>&1 | tee exp/scores/pool_results_xvector_snorm

# EER 5.80, DCF16 0.485
python3 ${scoring_dir}/scoring.py \
  ${scoring_dir}/eval_yue_config.ini -o exp/scores/converted_sre16_eval_scores_xvector_snorm 2>&1 | tee exp/scores/yue_results_xvector_snorm

# EER 14.66, DCF16 0.768
python3 ${scoring_dir}/scoring.py \
  ${scoring_dir}/eval_tgl_config.ini -o exp/scores/converted_sre16_eval_scores_xvector_snorm 2>&1 | tee exp/scores/tgl_results_xvector_snorm

echo "$0: results for fusion of ivectors and combined embeddings"
# Combine the scores from the combined embeddings and the ivectors
python local/combine_scores.py 1.0 exp/scores/sre16_eval_scores_xvector_snorm exp/scores/sre16_eval_scores_snorm > exp/scores/sre16_eval_scores_combined_snorm

echo "modelid    segment    side    LLR" > exp/scores/converted_sre16_eval_scores_combined_snorm
awk '{print $1, $2, "a", $3}' exp/scores/sre16_eval_scores_combined_snorm >> exp/scores/converted_sre16_eval_scores_combined_snorm

# EER 9.21, DCF16 0.575
python3 ${scoring_dir}/scoring.py \
  ${scoring_dir}/eval_config.ini -o exp/scores/converted_sre16_eval_scores_combined_snorm 2>&1 | tee exp/scores/pool_results_combined_snorm

# EER 5.08, DCF16 0.413
python3 ${scoring_dir}/scoring.py \
  ${scoring_dir}/eval_yue_config.ini -o exp/scores/converted_sre16_eval_scores_combined_snorm 2>&1 | tee exp/scores/yue_results_combined_snorm

# EER 12.92, DCF16 0.732
python3 ${scoring_dir}/scoring.py \
  ${scoring_dir}/eval_tgl_config.ini -o exp/scores/converted_sre16_eval_scores_combined_snorm 2>&1 | tee exp/scores/tgl_results_combined_snorm
fi
