#!/bin/bash

# TODO
# Get the number of frames for some features.  This is needed when
# getting egs.

nj=30
cmd="run.pl"
stage=0

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
if [ $# != 2 ]; then
  echo "Usage: $0 <in-data-dir> <feat-dir>"
  echo "TODO"
  exit 1;

fi

data_in=$1
dir=$2

name=`basename $data_in`

for f in $data_in/feats.scp ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log
featdir=${PWD}/$dir

sdata_in=$data_in/split$nj;
utils/split_data.sh $data_in $nj || exit 1;

$cmd JOB=1:$nj $dir/log/get_len_${name}.JOB.log \
  feat-to-len scp:${sdata_in}/JOB/feats.scp ark,t:$dir/utt2len_${name}.JOB || exit 1;

for n in $(seq $nj); do
  cat $dir/utt2len_${name}.$n || exit 1;
done > $dir/utt2len || exit 1

sort $dir/utt2len > ${data_in}/utt2len
