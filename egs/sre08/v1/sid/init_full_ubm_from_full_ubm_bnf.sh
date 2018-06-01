#!/bin/bash
# Copyright 2015   David Snyder
#           2015   Johns Hopkins University (Author: Daniel Garcia-Romero)
#           2015   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# This script derives a full-covariance UBM from DNN posteriors and
# speaker recognition features.

# Begin configuration section.
nj=40
cmd="run.pl"
stage=-2
delta_window=3
delta_order=2
subsample=5
min_post=0.025 # Minimum posterior to use (posteriors below this are pruned out)
posterior_scale=1.0 # This scale helps to control for successve features being highly
norm_vars=true
num_gselect=1000
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: steps/init_full_ubm_from_full_ubm.sh <data-speaker-id> <data-full-ubm> <ubm-model> <new-ubm-dir>"
  echo "Initializes a full-covariance UBM from DNN posteriors and speaker recognition features."
  echo " e.g.: steps/init_full_ubm_from_full_ubm.sh data/train data/train_bnf exp/full_ubm_bnf/final.mdl exp/full_ubm"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|16>                                      # number of parallel training jobs"
  echo "  --delta-window <n|3>                             # delta window size"
  echo "  --delta-order <n|2>                              # delta order"
  echo "                                                   # to be equal to the size of the DNN output layer."
  exit 1;
fi

data=$1
data_bnf=$2
in_dir=$3
dir=$4


for f in $data/feats.scp $data/vad.scp ${data_bnf}/feats.scp \
    ${in_dir}/final.ubm ${in_dir}/final.dubm; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj;
utils/split_data.sh $data $nj || exit 1;
sdata_bnf=$data_bnf/split$nj;
utils/split_data.sh $data_bnf $nj || exit 1;

delta_opts=`cat $in_dir/delta_opts 2>/dev/null`
if [ -f $in_dir/delta_opts ]; then
  cp $in_dir/delta_opts $dir/ 2>/dev/null
fi

logdir=$dir/log

feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- | apply-cmvn-sliding --norm-vars=${norm_vars} --center=true --cmn-window=300 ark:- ark:- | select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- | subsample-feats --n=$subsample ark:- ark:- |"

feats_bnf="ark,s,cs:apply-cmvn-sliding --norm-vars=${norm_vars} --center=true --cmn-window=300 scp:${sdata_bnf}/JOB/feats.scp ark:- | select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- | subsample-feats --n=$subsample ark:- ark:- |"

feats_sid="ark,s,cs:add-deltas --delta-order=2 --delta-window=3  scp:$sdata/JOB/feats.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- | select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- | subsample-feats --n=$subsample ark:- ark:- |"

num_components=2048

$cmd JOB=1:$nj $logdir/gselect.JOB.log \
  append-feats "$feats" "$feats_bnf" ark:- \| \
  gmm-gselect --n=$num_gselect ${in_dir}/final.dubm ark,s,cs:- \
  "ark:|gzip -c >$dir/gselect.JOB.gz" || exit 1;

$cmd JOB=1:$nj $logdir/make_stats.JOB.log \
  append-feats "$feats" "$feats_bnf" ark:- \| \
  fgmm-global-gselect-to-post ${in_dir}/final.ubm ark:- \
  "ark,s,cs:gunzip -c $dir/gselect.JOB.gz|"  ark:- \| \
  scale-post ark:- $posterior_scale ark:- \| \
  fgmm-global-acc-stats-post ark:- $num_components "$feats_sid" \
  $dir/stats.JOB.acc || exit 1;

$cmd $dir/log/init.log \
  fgmm-global-init-from-accs --verbose=2 \
  "fgmm-global-sum-accs - $dir/stats.*.acc |" $num_components \
  $dir/final.ubm || exit 1;

exit 0;
