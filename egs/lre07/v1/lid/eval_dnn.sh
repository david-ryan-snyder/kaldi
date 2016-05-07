#!/bin/bash
# Copyright  2016   David Snyder
# Apache 2.0.
#
# This script evaluates an LID DNN on the NIST LRE07 closed-set
# evaluation.

. cmd.sh
. path.sh
set -e

languages=local/general_lr_closed_set_langs.txt
cmd="run.pl"
use_gpu=no
chunk_size=1000
min_chunk_size=500
nj=6
stage=-1

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 <dnn-model> <data> <output-dir>"
fi
nnet=$1
data=$2
dir=$3
utt2lang=$data/utt2lang

mkdir -p $dir/log

classes="ark:lid/remove_dialect.pl $utt2lang \
         | utils/sym2int.pl -f 2 $languages - |"

utils/split_data.sh $data $nj
sdata=$data/split$nj/

if [ $stage -le 0 ]; then
  for i in `seq 1 $nj`; do
    $cmd $dir/compute.$i.log nnet3-xvector-compute-lid \
    --chunk-size=$chunk_size --min-chunk-size=$min_chunk_size \
    --use-gpu=$use_gpu $nnet scp:$sdata/$i/feats.scp ark,t:$dir/post.$i.vec || exit 1 &
  done
  wait;
cat $dir/post.1.vec > $dir/post.vec
for i in `seq 2 $nj`; do
  cat $dir/post.$i.vec >> $dir/post.vec
done
fi

if [ $stage -le 1 ]; then
  cat $dir/post.vec | \
    awk '{max=$3; argmax=3; for(f=3;f<NF;f++) { if ($f>max)
                            { max=$f; argmax=f; }}
                            print $1, (argmax - 3); }' | \
    utils/int2sym.pl -f 2 $languages \
      >$dir/output
fi

compute-wer --text ark:<(lid/remove_dialect.pl $utt2lang) \
  ark:$dir/output
