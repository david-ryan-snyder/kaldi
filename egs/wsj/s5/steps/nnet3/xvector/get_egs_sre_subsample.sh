#!/bin/bash

# TODO

# Copyright 2012-2016 Johns Hopkins University (Author: Daniel Povey)
#                2016 David Snyder
# Apache 2.0
#
# This script dumps training examples (egs) for xvector training.  These egs
# have only an input and no outputs (the inputs are typically MFCCs).  The egs
# consist of pairs of data chunks, with each chunk coming from different parts
# of the same utterance.  The two data-chunks in each eg will have respectively
# n=0 and n=1.  Each archive of egs has (in general) a different configuration,
# where a configuration is a pair of lengths of data-chunk (one for n=0 and one
# for n=1).  We don't mix together different lengths in the same archive,
# because it would require us to repeatedly run the compilation process within
# the same training job.
#
# This script, which will generally be called from other neural-net training
# scripts, extracts the training examples used to train the neural net (and also
# the validation examples used for diagnostics), and puts them in separate archives.


# Begin configuration section.
cmd=run.pl
# each archive has two sets of data-chunks,
# each of length randomly chosen between $min_frames_per_eg and $max_frames_per_eg.
# (however the lengths do not differ within the archives, to avoid triggering
# excessive recompilation of computation graphs).
subsample=5
min_frames_per_chunk=50
max_frames_per_chunk=300
frames_per_iter=10000000 # have this many frames per archive.
frame_shift=0.01

frames_per_iter_diagnostic=1000000 # have this many frames per archive for
                                   # the archives used for diagnostics.

num_diagnostic_archives=3  # we want to test the training and validation likelihoods
                           # on a range of utterance lengths, and this number controls
                           # how many archives we evaluate on.


compress=true   # set this to false to disable compression (e.g. if you want to see whether
                # results are affected).

num_heldout_spks=50     # number of utterances held out for validation.

num_repeats=1

stage=0
nj=6         # This should be set to the maximum number of jobs you are
             # comfortable to run in parallel; you can increase it if your disk
             # speed is greater and you have more machines.
stage=0

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 [opts] <data> <egs-dir>"
  echo " e.g.: $0 data/train exp/xvector_a/egs"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --nj <nj>                                        # The maximum number of jobs you want to run in"
  echo "                                                   # parallel (increase this only if you have good disk and"
  echo "                                                   # network speed).  default=6"
  echo "  --cmd (utils/run.pl;utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --min-frames-per-eg <#frames;50>                 # The minimum numer of frames per chunk that we dump"
  echo "  --max-frames-per-eg <#frames;200>                # The maximum numer of frames per chunk that we dump"
  echo "  --num-repeats <#repeats;1>                       # The (approximate) number of times the training"
  echo "                                                   # data is repeated in the egs"
  echo "  --frames-per-iter <#samples;1000000>             # Target number of frames per archive"
  echo "  --num-diagnostic-archives <#archives;3>          # Option that controls how many different versions of"
  echo "                                                   # the train and validation archives we create (e.g."
  echo "                                                   # train_subset.{1,2,3}.egs and valid.{1,2,3}.egs by default;"
  echo "                                                   # they contain different utterance lengths."
  echo "  --frames-per-iter-diagnostic <#samples;100000>   # Target number of frames for the diagnostic archives"
  echo "                                                   # {train_subset,valid}.*.egs"
  echo "  --stage <stage|0>                                # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."

  exit 1;
fi

data=$1
dir=$2

if [ ! -f $data/feats.scp ]; then
  echo "$0: expected $data/feats.scp to exist"
  exit 1
fi

feat_dim=$(feat-to-dim scp:$data/feats.scp -) || exit 1

mkdir -p $dir/info $dir/info $dir/temp
temp=$dir/temp

echo $feat_dim > $dir/info/feat_dim

if [ $stage -le 0 ]; then
  echo "$0: getting utt2len file"
  # note: this utt2len file is only an approximation of the number of
  # frames in each file.
  cp $data/utt2len $dir/temp/utt2len
fi

# Just supporting raw features.
feats="scp,s,cs:utils/filter_scp.pl $temp/uttlist.train $data/feats.scp |"
valid_feats="scp,s,cs:utils/filter_scp.pl $temp/uttlist.valid $data/feats.scp |"
train_subset_feats="scp,s,cs:utils/filter_scp.pl $temp/uttlist.train_subset $data/feats.scp |"

if [ $stage -le 1 ]; then
  echo "$0: getting list of validation utterances"

# Get list of validation utterances.
  utils/shuffle_list.pl $data/spk2utt | head -$num_heldout_spks > $temp/spk2utt.valid || exit 1;
  utils/spk2utt_to_utt2spk.pl $temp/spk2utt.valid > $temp/utt2spk.valid || exit 1;
  awk '{print $1}' $temp/utt2spk.valid > $temp/uttlist.valid || exit 1;
  awk '{print $1}' $temp/spk2utt.valid > $temp/spklist.valid || exit 1;
  utils/filter_scp.pl $temp/uttlist.valid <$temp/utt2len > $temp/utt2len.valid || exit 1;
  utils/filter_scp.pl $temp/uttlist.valid <$data/utt2reco > $temp/utt2reco.valid || exit 1;

  utils/filter_scp.pl --exclude $temp/spklist.valid $data/spk2utt | \
  utils/shuffle_list.pl | head -$num_heldout_spks > $temp/spk2utt.train_subset || exit 1;
  utils/spk2utt_to_utt2spk.pl $temp/spk2utt.train_subset > $temp/utt2spk.train_subset || exit 1;
  awk '{print $1}' $temp/utt2spk.train_subset > $temp/uttlist.train_subset || exit 1;
  awk '{print $1}' $temp/spk2utt.train_subset > $temp/spklist.train_subset || exit 1;
  utils/filter_scp.pl $temp/uttlist.train_subset <$temp/utt2len > $temp/utt2len.train_subset || exit 1;
  utils/filter_scp.pl $temp/uttlist.train_subset <$data/utt2reco > $temp/utt2reco.train_subset || exit 1;

  utils/filter_scp.pl --exclude $temp/uttlist.valid $temp/utt2len > $temp/utt2len.train || exit 1;
  utils/filter_scp.pl --exclude $temp/spklist.valid $data/spk2utt > $temp/spk2utt.train || exit 1;
  utils/filter_scp.pl --exclude $temp/uttlist.valid $data/utt2reco > $temp/utt2reco.train || exit 1;
  utils/spk2utt_to_utt2spk.pl $temp/spk2utt.train > $temp/utt2spk.train || exit 1;
  awk '{print $1}' $temp/utt2spk.train > $temp/uttlist.train || exit 1;
  awk '{print $1}' $temp/spk2utt.train > $temp/spklist.train || exit 1;
fi

#fi


# first for the training data... work out how many archives.

num_train_frames=$(awk '{n += $2} END{print n}' <$temp/utt2len.train)
num_valid_frames=$(awk '{n += $2} END{print n}' <$temp/utt2len.valid)
num_train_subset_frames=$(awk '{n += $2} END{print n}' <$temp/utt2len.train_subset)

echo $num_train_frames >$dir/info/num_frames

num_train_archives=$[($num_train_frames*$num_repeats)/$frames_per_iter + 1]
echo "$0: producing $num_train_archives archives for training"
echo $num_train_archives > $dir/info/num_archives
echo $num_diagnostic_archives > $dir/info/num_diagnostic_archives

#if [ 0 = 1 ]; then
if [ $nj -gt $num_train_archives ]; then
  echo "$0: reducing num-jobs $nj to number of training archives $num_train_archives"
  nj=$num_train_archives
fi

if [ $stage -le 2 ]; then
  if [ -e $dir/storage ]; then
    # Make soft links to storage directories, if distributing this way..  See
    # utils/create_split_dir.pl.
    echo "$0: creating data links"
    utils/create_data_link.pl $(for x in $(seq $num_train_archives); do echo $dir/egs.$x.ark; done)
    utils/create_data_link.pl $(for x in $(seq $num_train_archives); do echo $dir/egs_temp.$x.ark; done)
  fi
fi

if [ $stage -le 3 ]; then

  #if [ 0 = 1 ]; then
  echo "$0: allocating training examples"
  $cmd $dir/log/allocate_examples_train.log \
    steps/nnet3/xvector/allocate_examples_sre.py \
      --min-frames-per-chunk=$min_frames_per_chunk \
      --max-frames-per-chunk=$max_frames_per_chunk \
      --frames-per-iter=$frames_per_iter \
      --num-archives=$num_train_archives --num-jobs=$nj \
      $dir/temp/utt2len.train \
      $dir/temp/utt2reco.train \
      $dir/temp/utt2spk.train \
      $dir  || exit 1
  #fi

  echo "$0: allocating training subset examples"
  $cmd $dir/log/allocate_examples_train_subset.log \
    steps/nnet3/xvector/allocate_examples_sre.py \
      --prefix train_subset \
      --min-frames-per-chunk=$min_frames_per_chunk \
      --max-frames-per-chunk=$max_frames_per_chunk \
      --randomize-chunk-length false \
      --frames-per-iter=$frames_per_iter_diagnostic \
      --num-archives=$num_diagnostic_archives --num-jobs=1 \
      $dir/temp/utt2len.train_subset \
      $dir/temp/utt2reco.train_subset \
      $dir/temp/utt2spk.train_subset \
      $dir  || exit 1

  echo "$0: allocating validation examples"
  $cmd $dir/log/allocate_examples_valid.log \
    steps/nnet3/xvector/allocate_examples_sre.py \
      --prefix valid \
      --min-frames-per-chunk=$min_frames_per_chunk \
      --max-frames-per-chunk=$max_frames_per_chunk \
      --randomize-chunk-length false \
      --frames-per-iter=$frames_per_iter_diagnostic \
      --frames-per-iter=$frames_per_iter_diagnostic \
      --num-archives=$num_diagnostic_archives --num-jobs=1 \
      $dir/temp/utt2len.valid \
      $dir/temp/utt2reco.valid \
      $dir/temp/utt2spk.valid \
      $dir  || exit 1
#fi
fi

if [ $stage -le 4 ]; then
  echo "$0: Generating training examples on disk"
  rm $dir/.error 2>/dev/null
  #if [ 0 = 1 ]; then
  for g in $(seq $nj); do
    outputs=`awk '{for(i=1;i<=NF;i++)printf("ark:%s ",$i);}' $temp/outputs.$g`
    $cmd $dir/log/train_create_examples.$g.log \
      nnet3-xvector-get-egs-sre-subsample --subsample=$subsample $temp/ranges.$g \
      "`echo $feats | sed s/JOB/$g/g`" $outputs || touch $dir/.error &
  done
  #fi

  train_subset_outputs=`awk '{for(i=1;i<=NF;i++)printf("ark:%s ",$i);}' $temp/train_subset_outputs.1`
  echo "$0: Generating training subset examples on disk"
  $cmd $dir/log/train_subset_create_examples.1.log \
    nnet3-xvector-get-egs-sre-subsample --subsample=$subsample $temp/train_subset_ranges.1 \
    "$train_subset_feats" $train_subset_outputs || touch $dir/.error &

  valid_outputs=`awk '{for(i=1;i<=NF;i++)printf("ark:%s ",$i);}' $temp/valid_outputs.1`
  echo "$0: Generating validation examples on disk"
  $cmd $dir/log/valid_create_examples.1.log \
    nnet3-xvector-get-egs-sre-subsample --subsample=$subsample $temp/valid_ranges.1 \
    "$valid_feats" $valid_outputs || touch $dir/.error &

  wait
  if [ -f $dir/.error ]; then
    echo "$0: problem detected while dumping examples"
    exit 1
  fi
fi
#fi

if [ $stage -le 5 ]; then
  echo "$0: Shuffling order of archives on disk"
  num_batches=$(perl -e "print int($num_train_archives/$nj);")
  echo "num batches = $num_batches"
  echo "num training archives = $num_train_archives"
  for i in `seq 0 $num_batches`; do
    start_batch=$(perl -e "print int($i*$nj + 1);")
    tmp_end_batch=$(perl -e "print int($start_batch + $nj - 1);")
    end_batch=$(perl -e "print int($tmp_end_batch < $num_train_archives ? $tmp_end_batch : $num_train_archives);")
    $cmd JOB=$start_batch:$end_batch $dir/log/shuffle.JOB.log \
      nnet3-shuffle-egs --srand=JOB ark:$dir/egs_temp.JOB.ark ark:$dir/egs.JOB.ark  || exit 1;
  done

  $cmd JOB=1:$num_diagnostic_archives $dir/log/train_subset_shuffle.JOB.log \
    nnet3-shuffle-egs --srand=JOB ark:$dir/train_subset_egs_temp.JOB.ark ark:$dir/train_diagnostic_egs.JOB.ark  || exit 1;
  $cmd JOB=1:$num_diagnostic_archives $dir/log/valid_shuffle.JOB.log \
    nnet3-shuffle-egs --srand=JOB ark:$dir/valid_egs_temp.JOB.ark ark:$dir/valid_diagnostic_egs.JOB.ark  || exit 1;
fi

if [ $stage -le 6 ]; then
   for file in $(for x in $(seq $num_diagnostic_archives); do echo $dir/{valid,train_subset}_egs_temp.$x.ark; done) \
            $(for x in $(seq $num_train_archives); do echo $dir/egs_temp.$x.ark; done); do
     [ -L $file ] && rm $(readlink -f $file)
     rm $file
   done
fi

echo "$0: Finished preparing training examples"
