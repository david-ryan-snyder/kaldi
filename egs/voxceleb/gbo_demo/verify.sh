#!/bin/bash
. ./cmd.sh
. ./path.sh
train_cmd=run.pl
nnet_dir=exp/xvector_dgr5
#t=-10.2747
t=30
enroll=$1
echo "$enroll foo" > trial
$train_cmd $nnet_dir/scores/log/test.log \
  ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$nnet_dir/xvectors_enrollments/spk_num_utts.ark \
  "ivector-copy-plda --smoothing=0.0 $nnet_dir/xvectors_voxceleb_raw_combined_fast/plda - |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_voxceleb_raw_combined_fast/mean.vec scp:$nnet_dir/xvectors_enrollments/spk_xvector.scp ark:- | transform-vec $nnet_dir/xvectors_voxceleb_raw_combined_fast/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_voxceleb_raw_combined_fast/mean.vec scp:$nnet_dir/xvectors_test/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_voxceleb_raw_combined_fast/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" trial scores || exit 1;
score=`awk '{print $3}' scores`
if [ $(echo $score'>'$t | bc -l) -eq 1 ]; then
  echo "ACCEPT:$score";
else
  echo "REJECT:$score";
fi
