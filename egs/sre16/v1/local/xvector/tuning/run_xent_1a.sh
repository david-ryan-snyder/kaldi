#!/bin/bash

# This script trains a DNN similar to the recipe described in
# http://www.danielpovey.com/files/2017_interspeech_embeddings.pdf .

. ./cmd.sh
set -e

stage=1
train_stage=-10
use_gpu=true

data=data/swbd_sre_no_sil
nnet_dir=exp/xvector_xent_1a
egs_dir=exp/xvector_xent_1a/egs

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

num_pdfs=$(awk '{print $2}' $data/utt2spk | sort | uniq -c | wc -l)

# Currently, steps/nnet3/xvector/get_egs_xent.sh script is a little hard to
# deal with.  In the future, it will be redesigned.  For now,  the argument
# --num-repeats specifies the number of times a speaker repeats per archive.
# However, the number of archives is a function of all of the other arguments
# as well as the number of speakers and frames in your training data, so it
# can be hard to predict in advance how many training examples will be
# generated per speaker across all archives.
#
# To make sense of the egs script, I recommend putting an "exit 1" command in
# it immediately after stage 3.  Then, inspect exp/<your-dir>/egs/temp/ranges.* .
# The ranges files specify the examples that will be created, and which
# archives they will be stored in.  Each line of ranges.* has the following
# form:
#    <utt-id> <local-ark-indx> <global-ark-indx> <start-frame> <end-frame> <spk-id>
# For example:
#    100304-f-sre2006-kacg-A 1 2 4079 881 23

# If you're satisfied with the number of archives (e.g., 25-150 archives is
# reasonable) and with the number of examples per speaker (e.g., 500-5000
# is reasonable) then you can let the script continue to the later stages.
# Otherwise, try increasing or decreasing the --num-repeats option.  Rarely,
# you might need to fiddle with --frames-per-iter (e.g., if there are too many
# or too few examples in individual archives.
if [ $stage -le 4 ]; then
  echo "$0: getting neural network training egs";
  # dump egs.
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b{03,11,12,13}/$USER/kaldi-data/egs/sre16/v2/xent-$(date +'%m_%d_%H_%M')/$egs_dir/storage $egs_dir/storage
  fi
  steps/nnet3/xvector/get_egs_xent.sh --cmd "$train_cmd" \
    --nj 8 \
    --stage -3 \
    --frames-per-iter 500000000 \
    --frames-per-iter-diagnostic 100000 \
    --min-frames-per-chunk 200 \
    --max-frames-per-chunk 300 \
    --num-diagnostic-archives 3 \
    --num-repeats 37 \
    "$data" $egs_dir
fi

if [ $stage -le 5 ]; then
  echo "$0: creating neural net configs using the xconfig parser";
  num_targets=$(wc -w $egs_dir/pdf2num | awk '{print $1}')
  feat_dim=$(cat $egs_dir/info/feat_dim)
  mkdir -p $nnet_dir/configs
  cat <<EOF > $nnet_dir/configs/network.xconfig
  # please note that it is important to have input layer with the name=input

  # The frame-level layers
  input dim=$feat_dim name=input
  relu-renorm-layer name=tdnn1 input=Append(-2,-1,0,1,2) dim=512
  relu-renorm-layer name=tdnn2 input=Append(-2,0,2) dim=512
  relu-renorm-layer name=tdnn3 input=Append(-3,0,3) dim=512
  relu-renorm-layer name=tdnn4 dim=512
  relu-renorm-layer name=tdnn5 dim=1500

  # The stats pooling layer. Layers after this are segment-level.
  # In the config below, the first and last argument (0, and 10000)
  # means that we pool over an input segment starting at frame 0
  # and ending at frame 100000 or earlier.  The other arguments (1:1)
  # mean that no subsampling is performed.
  stats-layer name=stats config=mean+stddev(0:1:1:10000)

  # NOTE: the following layer wasn't a renorm layer in the INTERSPEECH paper.
  # We should try it with renormalization.  This is where we extract one
  # embedding from.
  relu-layer name=tdnn6 dim=512 input=stats

  # This is where another embedding is extracted from.
  relu-renorm-layer name=tdnn7 dim=300
  output-layer name=output include-log-softmax=true dim=$num_targets
EOF

  steps/nnet3/xconfig_to_configs.py \
      --xconfig-file $nnet_dir/configs/network.xconfig \
      --config-dir $nnet_dir/configs/
  cp $nnet_dir/configs/final.config $nnet_dir/nnet.config
fi


if [ $stage -le 6 ]; then
  # training for 4 epochs * 3 shifts means we see each eg 12
  # times (3 different frame-shifts of the same eg are counted as different).
  steps/nnet3/xvector/train_xent.sh --cmd "$train_cmd" \
      --cpu-cmd "$train_cmd" \
      --initial-effective-lrate 0.0025 \
      --final-effective-lrate 0.00025 \
      --diagnostic-period 10 \
      --max-param-change 0.75 \
      --minibatch-size 64 \
      --num-epochs 3 --num-shifts 1 --use-gpu $use_gpu --stage $train_stage \
      --num-jobs-initial 1 --num-jobs-final 4 \
      --egs-dir $egs_dir \
      $nnet_dir
fi

if [ $stage -le 7 ]; then
  # uncomment the following line to have it remove the egs when you are done.
  # steps/nnet2/remove_egs.sh $xvector_dir/egs
  exit 0;
fi

exit 0;
