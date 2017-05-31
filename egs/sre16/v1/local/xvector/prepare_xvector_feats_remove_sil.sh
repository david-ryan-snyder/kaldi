#!/bin/bash

# TODO
# This demonstrates how to create some standard features to train
# an x-vector system

nj=30
cmd="run.pl"
stage=0
norm_vars=false
center=true
left_context=4
right_context=4
cmn_window=300

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
if [ $# != 3 ]; then
  echo "Usage: $0 <in-data-dir> <out-data-dir> <feat-dir>"
  echo "TODO"
  exit 1;

fi

data_in=$1
data_out=$2
dir=$3

name=`basename $data_in`

for f in $data_in/feats.scp $data_in/vad.scp ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log
mkdir -p $data_out
featdir=${PWD}/$dir

cp $data_in/utt2spk $data_out/utt2spk
cp $data_in/spk2utt $data_out/spk2utt
cp $data_in/wav.scp $data_out/wav.scp

sdata_in=$data_in/split$nj;
utils/split_data.sh $data_in $nj || exit 1;

$cmd JOB=1:$nj $dir/log/create_xvector_feats_${name}.JOB.log \
  select-voiced-frames scp:${sdata_in}/JOB/feats.scp scp,s,cs:${sdata_in}/JOB/vad.scp \
  ark,scp:$featdir/xvector_feats_${name}.JOB.ark,$featdir/xvector_feats_${name}.JOB.scp || exit 1;

for n in $(seq $nj); do
  cat $featdir/xvector_feats_${name}.$n.scp || exit 1;
done > ${data_out}/feats.scp || exit 1
