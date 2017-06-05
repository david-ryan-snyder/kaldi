#!/bin/bash
# Copyright 2017   David Snyder
# Apache 2.0.
#
# See README.txt for more info on data required.

set -e

data_dir=$1

wget -P data/local/ http://www.openslr.org/resources/15/speaker_list.tgz
tar -C data/local/ -xvf data/local/speaker_list.tgz
sre_ref=data/local/speaker_list

local/make_sre.pl /export/corpora5/LDC/LDC2006S44/ \
   sre2004 $sre_ref $data_dir/sre2004

local/make_sre.pl /export/corpora5/LDC/LDC2011S01 \
  sre2005 $sre_ref $data_dir/sre2005_train

local/make_sre.pl /export/corpora5/LDC/LDC2011S04 \
  sre2005 $sre_ref $data_dir/sre2005_test

local/make_sre.pl /export/corpora5/LDC/LDC2011S09 \
  sre2006 $sre_ref $data_dir/sre2006_train

local/make_sre.pl /export/corpora5/LDC/LDC2011S10 \
  sre2006 $sre_ref $data_dir/sre2006_test_1

local/make_sre.pl /export/corpora5/LDC/LDC2012S01 \
  sre2006 $sre_ref $data_dir/sre2006_test_2
rm data/local/speaker_list.*
