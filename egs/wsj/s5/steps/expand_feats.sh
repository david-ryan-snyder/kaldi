#!/bin/bash

# Copyright 2016 David Snyder

# Apache 2.0

# Begin configuration section.
cmd=run.pl
nj=4
min_length=100
expand_type="zero" # zero: pads the features with zeros on the left and right
                   # tile: copies the entire feature matrix repeatedly
                   # copy: pads the features with copies of the first and last frame

# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 3 ]; then
   echo "usage: $0 [options] <src-data-dir> <log-dir> <path-to-storage-dir>";
   echo "options: "
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
logdir=$2
dir=$3

# make $mfccdir an absolute pathname.
dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $dir ${PWD}`

mkdir -p $dir || exit 1;
mkdir -p $logdir || exit 1;

utils/split_data.sh $data $nj || exit 1;
sdata=$data/split$nj

# use "name" as part of name of the archive.
name=`basename $data`

$cmd JOB=1:$nj $logdir/${name}_expand.JOB.log \
   expand-feats --min-feat=$min_length --type=$expand_type \
     scp:$sdata/JOB/feats.scp \
     ark,scp:$dir/${name}_expand.JOB.ark,$dir/${name}_expand.JOB.scp || exit 1;

# concatenate the .scp files together.
mv $data/feats.scp $data/feats.scp.bak
cat $dir/${name}_expand.*.scp > $data/feats.scp

utils/fix_data_dir.sh $data

echo "Succeeded expanding features for $name into $data"
