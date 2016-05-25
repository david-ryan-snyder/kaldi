#!/bin/bash
# Copyright  2016 David Snyder
# Apache 2.0.
#
# This script demonstrates training a DNN for the NIST LRE07 eval

. cmd.sh
. path.sh
set -e

mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
languages=local/general_lr_closed_set_langs.txt

# Training data sources
local/make_sre_2008_train.pl /export/corpora5/LDC/LDC2011S05 data
local/make_callfriend.pl /export/corpora/LDC/LDC96S60 vietnamese data
local/make_callfriend.pl /export/corpora/LDC/LDC96S59 tamil data
local/make_callfriend.pl /export/corpora/LDC/LDC96S53 japanese data
local/make_callfriend.pl /export/corpora/LDC/LDC96S52 hindi data
local/make_callfriend.pl /export/corpora/LDC/LDC96S51 german data
local/make_callfriend.pl /export/corpora/LDC/LDC96S50 farsi data
local/make_callfriend.pl /export/corpora5/LDC/LDC96S48 french data
local/make_callfriend.pl /export/corpora5/LDC/LDC96S49 arabic.standard data
local/make_callfriend.pl /export/corpora5/LDC/LDC96S54 korean data
local/make_callfriend.pl /export/corpora5/LDC/LDC96S55 chinese.mandarin.mainland data
local/make_callfriend.pl /export/corpora5/LDC/LDC96S56 chinese.mandarin.taiwan data
local/make_callfriend.pl /export/corpora5/LDC/LDC96S57 spanish.caribbean data
local/make_callfriend.pl /export/corpora5/LDC/LDC96S58 spanish.noncaribbean data
local/make_lre96.pl /export/corpora/NIST/lid96e1 data
local/make_lre03.pl /export/corpora4/LDC/LDC2006S31 data
local/make_lre05.pl /export/corpora5/LDC/LDC2008S05 data
local/make_lre07_train.pl /export/corpora5/LDC/LDC2009S05 data
local/make_lre09.pl /export/corpora5/NIST/LRE/LRE2009/eval data

# Make the evaluation data set. We're concentrating on the General Language
# Recognition Closed-Set evaluation, so we remove the dialects and filter
# out the unknown languages used in the open-set evaluation.
#local/make_lre07.pl /export/corpora5/LDC/LDC2009S04 data/lre07_all

local/make_lre07.pl /export/corpora5/LDC/LDC2009S04 data/lre07_all
cp -r data/lre07_all data/lre07
utils/filter_scp.pl -f 2 $languages <(lid/remove_dialect.pl data/lre07_all/utt2lang) \
  > data/lre07/utt2lang
utils/fix_data_dir.sh data/lre07

src_list="data/sre08_train_10sec_female \
    data/sre08_train_10sec_male data/sre08_train_3conv_female \
    data/sre08_train_3conv_male data/sre08_train_8conv_female \
    data/sre08_train_8conv_male data/sre08_train_short2_male \
    data/sre08_train_short2_female data/ldc96* data/lid05d1 \
    data/lid05e1 data/lid96d1 data/lid96e1 data/lre03 \
    data/ldc2009* data/lre09"

# Remove any spk2gender files that we have: since not all data
# sources have this info, it will cause problems with combine_data.sh
for d in $src_list; do rm -f $d/spk2gender 2>/dev/null; done

utils/combine_data.sh data/train_unsplit $src_list

# original utt2lang will remain in data/train_unsplit/.backup/utt2lang.
utils/apply_map.pl -f 2 --permissive local/lang_map.txt \
  < data/train_unsplit/utt2lang 2>/dev/null > foo
cp foo data/train_unsplit/utt2lang
rm foo

echo "**Language count for DNN training:**"
awk '{print $2}' data/train/utt2lang | sort | uniq -c | sort -nr

steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/train exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/lre07 exp/make_mfcc $mfccdir

lid/compute_vad_decision.sh --nj 10 --cmd "$train_cmd" data/train \
  exp/make_vad $vaddir
lid/compute_vad_decision.sh --nj 10 --cmd "$train_cmd" data/lre07 \
  exp/make_vad $vaddir

# NOTE: Example of removing the silence. In this case, the features
# are mean normlized MFCCs.
if [ 0 = 1 ]; then
nj=20
feats_dir=mfcc/feats_cmvn_no_sil
for data in lre07 train; do
  sdata=data/$data/split$nj;
  echo "making cmvn vad stats for $data"
  utils/split_data.sh data/$data $nj || exit 1;
  mkdir -p ${feats_dir}/log/
  cp -r data/${data} data/${data}_cmvn_no_sil
  queue.pl JOB=1:$nj ${feats_dir}/log/${data}_cmvn_no_sil.JOB.log \
    apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:$sdata/JOB/feats.scp ark:- \| \
    select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp \
    ark,scp:${feats_dir}/cmvn_no_sil_${data}_feats.JOB.ark,${feats_dir}/cmvn_no_sil_${data}_feats.JOB.scp || exit 1;
    utils/fix_data_dir.sh data/${data}_cmvn_no_sil
    echo "finished making cmvn vad stats for $data"
done
fi

# NOTE:
# This script will expand the feature matrices, as may be required by the DNN.
# It supports 3 ways of expanding: "tile" copies the entire feature matrix repeatedly,
# "zero" pads with 0 on the left and right, and "copy" pads by copying the first and last
# frames repeatedly. The option --min-length is the target number of frames. If an
# utterance has more frames than this, it is unmodified, otherwise, it is expanded to
# equal min-length.
# NOTE: This script is applied directly to the data directory; it does not make a
# copy (so make a copy first).
steps/expand_feats.sh --cmd "$train_cmd" --min-length 400 \
                      --expand-type "tile" \
                      --nj 40 \
                      data/lre07_cmvn_no_sil_expand \
                      exp/expand_feats $mfccdir
steps/expand_feats.sh --cmd "$train_cmd" --min-length 400 \
                      --expand-type "tile" \
                      --nj 40 \
                      data/train_cmvn_no_sil_expand \
                      exp/expand_feats $mfccdir

utils/fix_data_dir.sh data/lre07_cmvn_no_sil_expand
utils/fix_data_dir.sh data/train_cmvn_no_sil_expand

# NOTE: This script trains the DNN
local/xvector/run_lid.sh --train-stage -10 \
                         --stage -10 \
                         --data data/train_cmvn_no_sil_expand \
                         --nnet-dir exp/xvector_lid_a \
                         --egs-dir exp/xvector_lid_a/egs


# NOTE: Example script of how to extract posteriors from the DNN after it's
# trained. Also does an eval on lre07.
lid/eval_dnn.sh --cmd "$eval_cmd" --chunk-size 3000 \
                --min-chunk-size 500 --use-gpu yes \
                --nj 6 \
                exp/xvector_lid_a/900.raw data/lre07_cmvn_no_sil_expand \
                exp/lre07_results
