#!/bin/bash

#TODO

. ./cmd.sh
set -e

stage=1
train_stage=-10
use_gpu=true
feat_dim=180 # this is the MFCC dim we use in the hires features.  you can't change it
            # unless you change local/xvector/prepare_perturbed_data.sh to use a different
            # MFCC config with a different dimension.
data=data/xvector_sre # you can't change this without changing
                     # local/xvector/prepare_perturbed_data.sh
xvector_dim=300 # dimension of the xVector.  configurable.
xvector_dir=exp/xvector_sre_a1
egs_dir=exp/xvector_sre_a1/egs


. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

if [ $stage -le 3 ]; then
  # Prepare configs
  mkdir -p $xvector_dir/log

  $train_cmd $xvector_dir/log/make_configs.log \
    steps/nnet3/xvector/make_jesus_configs.py \
      --output-dim $xvector_dim \
      --splice-indexes="0 0 0 0 mean(0:3:6:6000)" \
      --feat-dim $feat_dim --output-dim $xvector_dim \
      --num-jesus-blocks 150 \
      --jesus-input-dim 600 --jesus-output-dim 3000 --jesus-hidden-dim 2000 \
      $xvector_dir/nnet.config
fi


if [ $stage -le 4 ]; then
  # dump egs.
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b{03,11,12,13}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5/$egs_dir/storage $egs_dir/storage
  fi

  # Params like num-repeats and frames-per-iter will need to be modified
  # based on the amount of data you have.
  steps/nnet3/xvector/get_egs_sre.sh --cmd "$train_cmd" \
    --nj 8 \
    --stage 0 \
    --frames-per-iter 2000000 \
    --frames-per-iter-diagnostic 200000 \
    --min-frames-per-chunk 1000 \
    --max-frames-per-chunk 3000 \
    --num-diagnostic-archives 1 \
    --num-repeats 4 \
    "$data" $egs_dir
fi

if [ $stage -le 5 ]; then
  # training for 4 epochs * 3 shifts means we see each eg 12
  # times (3 different frame-shifts of the same eg are counted as different).
  steps/nnet3/xvector/train.sh --cmd "$train_cmd" \
      --initial-effective-lrate 0.002 \
      --final-effective-lrate 0.0002 \
      --max-param-change 0.2 \
      --minibatch-size 16 \
      --num-epochs 4 --num-shifts 3 --use-gpu $use_gpu --stage $train_stage \
      --num-jobs-initial 1 --num-jobs-final 8 \
      --egs-dir $egs_dir \
      $xvector_dir
fi

if [ $stage -le 6 ]; then
  # uncomment the following line to have it remove the egs when you are done.
  # steps/nnet2/remove_egs.sh $xvector_dir/egs
fi


exit 0;