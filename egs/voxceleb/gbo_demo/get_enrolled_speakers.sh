#!/bin/bash
. ./cmd.sh
. ./path.sh
nnet_dir=exp/xvector_dgr5
awk '{print $1}' $nnet_dir/xvectors_enrollments/spk_xvector.scp | tr '\n' ';'
