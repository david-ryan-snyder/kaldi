#!/bin/bash

# Copyright      2017 Johns Hopkins University (Author: Daniel Povey)
#                2017 David Snyder
# Apache 2.0
#
# This script dumps training examples (egs) for xvector training.  These egs
# consist of a data chunk and a label. Each archive of egs has, in general,
# a different input chunk-size. We don't mix together different lengths in the same archive,
# because it would require us to repeatedly run the compilation process within
# the same training job.
#
# This script, which will generally be called from other neural-net training
# scripts, extracts the training examples used to train the neural net (and also
# the validation examples used for diagnostics), and puts them in separate archives.


# Begin configuration section.
cmd=run.pl
# each archive has data-chunks off length randomly chosen between
# $min_frames_per_eg and $max_frames_per_eg.
min_frames_per_chunk=50
max_frames_per_chunk=300
frames_per_iter=10000000 # target number of frames per archive.

frames_per_iter_diagnostic=100000 # have this many frames per archive for
                                   # the archives used for diagnostics.

num_diagnostic_archives=3  # we want to test the training likelihoods
                           # on a range of utterance lengths, and this number controls
                           # how many archives we evaluate on.


compress=true   # set this to false to disable compression (e.g. if you want to see whether
                # results are affected).

num_heldout_utts=100     # number of utterances held out for training subset

num_repeats=1 # number of times each speaker repeats per archive

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

if [ ! -f $data/utt2len ]; then
  feat-to-len scp:$data/feats.scp ark,t:$data/utt2len || exit 1;
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

if [ $stage -le 1 ]; then
  echo "$0: getting list of validation utterances"

  awk '{print $1}' $data/utt2spk | utils/shuffle_list.pl | head -$num_heldout_utts > $temp/train_subset_uttlist || exit 1;

  awk '{print $1}' $temp/utt2len |
  cp $temp/utt2len $temp/utt2len.train
  utils/filter_scp.pl $temp/train_subset_uttlist <$temp/utt2len > $temp/utt2len.train_subset

  python local/xvector/spk2utt_to_spk2int.py $data/spk2utt > $temp/spk2int
  utils/sym2int.pl -f 2 $temp/spk2int $data/utt2spk > $temp/utt2int
  utils/filter_scp.pl $temp/utt2len.train $temp/utt2int > $temp/utt2int.train
  utils/filter_scp.pl $temp/utt2len.train_subset $temp/utt2int > $temp/utt2int.train_subset
fi

num_pdfs=$(awk '{print $2}' $temp/utt2int | sort | uniq -c | wc -l)

# Just supporting raw features.
feats="scp,s,cs:utils/filter_scp.pl $temp/ranges.JOB $data/feats.scp |"
train_subset_feats="scp,s,cs:utils/filter_scp.pl $temp/train_subset_uttlist $data/feats.scp |"


# first for the training data... work out how many archives.

num_train_frames=$(awk '{n += $2} END{print n}' <$temp/utt2len.train)
num_train_subset_frames=$(awk '{n += $2} END{print n}' <$temp/utt2len.train_subset)

echo $num_train_frames >$dir/info/num_frames

num_train_archives=$[($num_train_frames*$num_repeats)/$frames_per_iter + 1]
echo "$0: producing $num_train_archives archives for training"
echo $num_train_archives > $dir/info/num_archives
echo $num_diagnostic_archives > $dir/info/num_diagnostic_archives


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
  echo "$0: allocating training examples"
  $cmd $dir/log/allocate_examples_train.log \
    steps/nnet3/xvector/allocate_examples_xent.py \
      --num-repeats=$num_repeats \
      --min-frames-per-chunk=$min_frames_per_chunk \
      --max-frames-per-chunk=$max_frames_per_chunk \
      --frames-per-iter=$frames_per_iter \
      --num-archives=$num_train_archives --num-jobs=$nj \
      $dir/temp/utt2len.train $dir/temp/utt2int.train $dir  || exit 1

  echo "$0: allocating training subset examples"
  $cmd $dir/log/allocate_examples_train_subset.log \
    steps/nnet3/xvector/allocate_examples_xent.py \
      --prefix train_subset \
      --num-repeats=1 \
      --min-frames-per-chunk=$min_frames_per_chunk \
      --max-frames-per-chunk=$max_frames_per_chunk \
      --randomize-chunk-length false \
      --frames-per-iter=$frames_per_iter_diagnostic \
      --num-archives=$num_diagnostic_archives --num-jobs=1 \
      $dir/temp/utt2len.train_subset $dir/temp/utt2int.train_subset $dir  || exit 1

fi

# Might want to put an exit 1 command here and look at exp/$dir/temp/ranges.* before deciding if you want to continue to the next stage.

if [ $stage -le 4 ]; then
  echo "$0: Generating training examples on disk"
  rm $dir/.error 2>/dev/null
  for g in $(seq $nj); do
    outputs=`awk '{for(i=1;i<=NF;i++)printf("ark:%s ",$i);}' $temp/outputs.$g`
    $cmd $dir/log/train_create_examples.$g.log \
      nnet3-xvector-get-egs-xent --num-pdfs=$num_pdfs $temp/ranges.$g \
      "`echo $feats | sed s/JOB/$g/g`" $outputs || touch $dir/.error &
  done
  train_subset_outputs=`awk '{for(i=1;i<=NF;i++)printf("ark:%s ",$i);}' $temp/train_subset_outputs.1`
  echo "$0: Generating training subset examples on disk"
  $cmd $dir/log/train_subset_create_examples.1.log \
    nnet3-xvector-get-egs-xent --num-pdfs=$num_pdfs $temp/train_subset_ranges.1 \
    "$train_subset_feats" $train_subset_outputs || touch $dir/.error &
  wait
  if [ -f $dir/.error ]; then
    echo "$0: problem detected while dumping examples"
    exit 1
  fi
fi

if [ $stage -le 5 ]; then
  echo "$0: Shuffling order of archives on disk"
  $cmd --max-jobs-run $nj JOB=1:$num_train_archives $dir/log/shuffle.JOB.log \
    nnet3-shuffle-egs --srand=JOB ark:$dir/egs_temp.JOB.ark ark:$dir/egs.JOB.ark  || exit 1;
  $cmd --max-jobs-run $nj JOB=1:$num_diagnostic_archives $dir/log/train_subset_shuffle.JOB.log \
    nnet3-shuffle-egs --srand=JOB ark:$dir/train_subset_egs_temp.JOB.ark ark:$dir/train_diagnostic_egs.JOB.ark  || exit 1;
fi

if [ $stage -le 6 ]; then
   for file in $(for x in $(seq $num_diagnostic_archives); do echo $dir/train_subset_egs_temp.$x.ark; done) \
            $(for x in $(seq $num_train_archives); do echo $dir/egs_temp.$x.ark; done); do
     [ -L $file ] && rm $(readlink -f $file)
     rm $file
   done
fi

echo "$0: Finished preparing training examples"
