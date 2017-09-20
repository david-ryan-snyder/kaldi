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

# SRE16 trials
sre16_trials=data/sre16_eval_test/trials
sre16_trials_tgl=data/sre16_eval_test/trials_tgl
sre16_trials_yue=data/sre16_eval_test/trials_yue

stage=0
if [ $stage -le 0 ]; then
  # Path to some, but not all of the training corpora
  data_root=/export/corpora/LDC

  # Prepare telephone and microphone speech from Mixer6.
  local/make_mx6.sh $data_root/LDC2013S03 data/

  # Prepare SRE10 test and enroll. Includes microphone interview speech.
  # NOTE: This corpus is now available through the LDC as LDC2017S06.
  local/make_sre10.pl /export/corpora5/SRE/SRE2010/eval/ data/

  # Prepare SRE08 test and enroll. Includes some microphone speech.
  local/make_sre08.pl $data_root/LDC2011S08 $data_root/LDC2011S05 data/

  # This prepares the older NIST SREs from 2004-2006.
  local/make_sre.sh $data_root data/

  # Combine all SREs prior to 2016 and Mixer6 into one dataset
  utils/combine_data.sh data/sre \
    data/sre2004 data/sre2005_train \
    data/sre2005_test data/sre2006_train \
    data/sre2006_test_1 data/sre2006_test_2 \
    data/sre08 data/mx6 data/sre10
  utils/validate_data_dir.sh --no-text --no-feats data/sre
  utils/fix_data_dir.sh data/sre

  # Prepare SWBD corpora.
  local/make_swbd_cellular1.pl $data_root/LDC2001S13 \
    data/swbd_cellular1_train
  local/make_swbd_cellular2.pl /export/corpora5/LDC/LDC2004S07 \
    data/swbd_cellular2_train
  local/make_swbd2_phase1.pl $data_root/LDC98S75 \
    data/swbd2_phase1_train
  local/make_swbd2_phase2.pl /export/corpora5/LDC/LDC99S79 \
    data/swbd2_phase2_train
  local/make_swbd2_phase3.pl /export/corpora5/LDC/LDC2002S06 \
    data/swbd2_phase3_train

  # Combine all SWB corpora into one dataset.
  utils/combine_data.sh data/swbd \
    data/swbd_cellular1_train data/swbd_cellular2_train \
    data/swbd2_phase1_train data/swbd2_phase2_train data/swbd2_phase3_train

  # Prepare NIST SRE 2016 evaluation data.
  local/make_sre16_eval.pl /export/corpora5/SRE/R149_0_1 data

  # Prepare unlabeled Cantonese and Tagalog development data. This dataset
  # was distributed to SRE participants.
  local/make_sre16_unlabeled.pl /export/corpora5/SRE/LDC2016E46_SRE16_Call_My_Net_Training_Data data
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in sre swbd sre16_eval_enroll sre16_eval_test sre16_major; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
fi

if [ $stage -le 2 ]; then
  # Train the UBM.
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 20G" \
    --nj 40 --num-threads 8  --subsample 1 \
    data/sre16_major 2048 \
    exp/diag_ubm

  sid/train_full_ubm.sh --cmd "$train_cmd --mem 25G" \
    --nj 40 --remove-low-count-gaussians false --subsample 1 \
    data/sre16_major \
    exp/diag_ubm exp/full_ubm
fi

if [ $stage -le 3 ]; then
  # Train the i-vector extractor.
  utils/combine_data.sh data/swbd_sre data/swbd data/sre
  sid/train_ivector_extractor.sh --cmd "$train_cmd --mem 35G" \
    --ivector-dim 600 \
    --num-iters 5 \
    exp/full_ubm/final.ubm data/swbd_sre \
    exp/extractor
fi

# In this section, we augment the SRE data with reverberation,
# noise, music, and babble, and combined it with the clean SRE
# data.  The combined list will be used to train the PLDA model.
if [ $stage -le 4 ]; then
  utils/data/get_utt2num_frames.sh --nj 40 --cmd "$train_cmd" data/sre
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/sre/utt2num_frames > data/sre/reco2dur

  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the SRE list.  Note that we don't add any
  # additive noise here.
  python steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 8000 \
    data/sre data/sre_reverb
  cp data/sre/vad.scp data/sre_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/sre_reverb data/sre_reverb.new
  rm -rf data/sre_reverb
  mv data/sre_reverb.new data/sre_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  local/make_musan.sh /export/corpora/JHU/musan data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  python steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/sre data/sre_noise
  # Augment with musan_music
  python steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/sre data/sre_music
  # Augment with musan_speech
  python steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/sre data/sre_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/sre_aug data/sre_reverb data/sre_noise data/sre_music data/sre_babble

  # Take a random subset of the augmentations (64k is roughly the size of the SRE dataset)
  utils/subset_data_dir.sh data/sre_aug 64000 data/sre_aug_64k
  utils/fix_data_dir.sh data/sre_aug_64k

  # Make MFCCs for the augmented data.  Note that we want we should alreay have the vad.scp
  # from the clean version at this point, which is identical to the clean version!
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    data/sre_aug_64k exp/make_mfcc $mfccdir

  # Combine the clean and augmented SRE list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/sre_combined data/sre_aug_64k data/sre
fi

if [ $stage -le 5 ]; then
  # Extract i-vectors for SRE data (includes Mixer 6). We'll use this for
  # things like LDA or PLDA.
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
    exp/extractor data/sre_combined \
    exp/ivectors_sre_combined

  # The SRE16 major is an unlabeled dataset consisting of Cantonese and
  # and Tagalog.  This is useful for things like centering, whitening and
  # score normalization.
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
    exp/extractor data/sre16_major \
    exp/ivectors_sre16_major

  # The SRE16 test data
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
    exp/extractor data/sre16_eval_test \
    exp/ivectors_sre16_eval_test

  # The SRE16 enroll data
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
    exp/extractor data/sre16_eval_enroll \
    exp/ivectors_sre16_eval_enroll
fi

if [ $stage -le 6 ]; then
  # Compute the mean vector for centering the evaluation i-vectors.
  $train_cmd exp/ivectors_sre16_major/log/compute_mean.log \
    ivector-mean scp:exp/ivectors_sre16_major/ivector.scp \
    exp/ivectors_sre16_major/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  $train_cmd exp/ivectors_sre_combined/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_sre_combined/ivector.scp ark:- |" \
    ark:data/sre_combined/utt2spk exp/ivectors_sre_combined/transform.mat || exit 1;

  #  Train the PLDA model.
  $train_cmd exp/ivectors_sre_combined/log/plda.log \
    ivector-compute-plda ark:data/sre_combined/spk2utt \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_sre_combined/ivector.scp ark:- | transform-vec exp/ivectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    exp/ivectors_sre_combined/plda || exit 1;
fi

if [ $stage -le 7 ]; then
  $train_cmd exp/scores/log/sre16_eval_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:exp/ivectors_sre16_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 exp/ivectors_sre_combined/plda - |" \
    "ark:ivector-mean ark:data/sre16_eval_enroll/spk2utt scp:exp/ivectors_sre16_eval_enroll/ivector.scp ark:- | ivector-subtract-global-mean exp/ivectors_sre16_major/mean.vec ark:- ark:- | transform-vec exp/ivectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/ivectors_sre16_major/mean.vec scp:exp/ivectors_sre16_eval_test/ivector.scp ark:- | transform-vec exp/ivectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre16_trials' | cut -d\  --fields=1,2 |" exp/scores/sre16_eval_scores || exit 1;

  pooled_eer=`compute-eer <(python local/prepare_for_eer.py $sre16_trials exp/scores/sre16_eval_scores)  2> /dev/null`
  tgl_eer=`compute-eer <(python local/prepare_for_eer.py $sre16_trials_tgl exp/scores/sre16_eval_scores)  2> /dev/null`
  yue_eer=`compute-eer <(python local/prepare_for_eer.py $sre16_trials_yue exp/scores/sre16_eval_scores)  2> /dev/null`
  echo "EER: Pooled ${pooled_eer}%, Tagalog ${tgl_eer}%, Cantonese ${yue_eer}%"
  # EER: Pooled 13.65%, Tagalog 17.73%, Cantonese 9.612%
fi