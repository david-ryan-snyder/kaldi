#!/bin/bash

#TODO

. ./cmd.sh
set -e

stage=1
train_stage=-10
generate_alignments=true # false if doing ctc training
use_gpu=true
feat_dim=7 # this is the MFCC dim we use in the hires features.  you can't change it
            # unless you change local/xvector/prepare_perturbed_data.sh to use a different
            # MFCC config with a different dimension.
#data=data/train_cmvn_no_sil  # you can't change this without changing
data=data/train_cmvn_no_sil
                                # local/xvector/prepare_perturbed_data.sh
nnet_dir=exp/xvector_lid_b
egs_dir=exp/xvector_lid_b/egs

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

num_pdfs=$(awk '{print $2}' $data/utt2lang | sort | uniq -c | wc -l)

#if [ 0 = 1 ]; then
if [ $stage -le 3 ]; then
  # Prepare configs
  mkdir -p $nnet_dir/log

  $train_cmd $nnet_dir/log/make_configs.log \
    steps/nnet3/xvector/make_jesus_configs_lid.py \
      --splice-indexes="-4,-3,-2,-1,0,1,2,3,4 -4,4 -8,8 -16,16 -32,32 -64,64 mean+stddev+count(0:8:64:8192)" \
      --feat-dim $feat_dim \
      --num-jesus-blocks 60 \
      --jesus-input-dim 600 --jesus-output-dim 600 --jesus-hidden-dim 6000 \
      --num-targets $num_pdfs \
      $nnet_dir/presoftmax_prior_scale.vec \
      $nnet_dir/nnet.config
fi
#fi

#if [ 0 = 1 ]; then
if [ $stage -le 4 ]; then
  # dump egs.
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b{03,11,12,13}/$USER/kaldi-data/egs/lre-$(date +'%m_%d_%H_%M')/v2/$egs_dir/storage $egs_dir/storage
  fi
  steps/nnet3/xvector/get_egs_lid.sh --cmd "$train_cmd" \
    --nj 8 \
    --stage -3 \
    --frames-per-iter 3000000 \
    --frames-per-iter-diagnostic 300000 \
    --min-frames-per-chunk 299 \
    --max-frames-per-chunk 3000 \
    --num-diagnostic-archives 3 \
    --num-repeats 3 \
    "$data" $egs_dir
fi
#fi

if [ $stage -le 5 ]; then
  # training for 4 epochs * 3 shifts means we see each eg 12
  # times (3 different frame-shifts of the same eg are counted as different).
  steps/nnet3/xvector/train_lid.sh --cmd "$train_cmd" \
      --cpu-cmd "$xvector_cmd" \
      --initial-effective-lrate 0.002 \
      --final-effective-lrate 0.0002 \
      --max-param-change 0.2 \
      --minibatch-size 32 \
      --num-epochs 8 --num-shifts 3 --use-gpu $use_gpu --stage $train_stage \
      --num-jobs-initial 1 --num-jobs-final 8 \
      --egs-dir $egs_dir \
      $nnet_dir
fi


exit 1;

if [ $stage -le 6 ]; then
  # uncomment the following line to have it remove the egs when you are done.
  # steps/nnet2/remove_egs.sh $xvector_dir/egs
fi


exit 0;
