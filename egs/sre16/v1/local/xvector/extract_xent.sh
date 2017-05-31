#!/bin/bash
# TODO
# Extract xvectors from two layers of a DNN trained using multiclass xent.

. cmd.sh
. path.sh
set -e

nj=100
cmd="run.pl"
chunk_size=1000
b_offset=512
b_dim=300
a_offset=0
a_dim=512
left_context=2
right_context=2

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 <nnet-dir> <data> <embedding-dir>"
  echo "TODO"
fi

srcdir=$1
data=$2
dir=$3

nnet="nnet3-init $srcdir/final.raw $srcdir/extract.config - |"

for f in $srcdir/final.raw $srcdir/extract.config $data/feats.scp ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

mkdir -p $dir/log

utils/split_data.sh $data $nj
echo "$0: extracting xvectors for $data"
datasplit=$data/split$nj/JOB

feat="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:$datasplit/feats.scp ark:- | select-voiced-frames-special --left-context=$left_context --right-context=$right_context ark:- scp,s,cs:${datasplit}/vad.scp ark:- |"

$cmd JOB=1:$nj ${dir}/log/extract.JOB.log \
  nnet3-xvector-compute-simple --use-gpu=no --chunk-size=$chunk_size \
    "$nnet" "$feat" ark,scp:${dir}/xvectors.JOB.ark,${dir}/xvectors.JOB.scp || exit 1;

echo "Extracted xvectors for $data"
cat ${dir}/xvectors.1.scp > ${dir}/xvectors.scp
for i in `seq 2 $nj`; do
  cat ${dir}/xvectors.$i.scp >> ${dir}/xvectors.scp
done

echo "Separating xvectors into A and B parts"
$cmd $dir/log/extract_a.log \
  select-subvectors --offset=$a_offset --dim=$a_dim \
    scp:${dir}/xvectors.scp ark,scp:${dir}/xvector_a.ark,${dir}/xvector_a.scp || exit 1;

$cmd $dir/log/extract_a.log \
  select-subvectors --offset=$b_offset --dim=$b_dim \
    scp:${dir}/xvectors.scp ark,scp:${dir}/xvector_b.ark,${dir}/xvector_b.scp || exit 1;

echo "Computing means for A and B parts"
$cmd ${dir}/log/compute_mean_a.log \
  ivector-mean scp:${dir}/xvector_a.scp \
   ${dir}/xvector_a_mean.vec || exit 1;

$cmd ${dir}/log/compute_mean_b.log \
  ivector-mean scp:${dir}/xvector_b.scp \
   ${dir}/xvector_b_mean.vec || exit 1;
