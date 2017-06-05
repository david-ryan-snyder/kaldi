#!/usr/bin/perl
#
# Copyright   2017   David Snyder
# Apache 2.0

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-LDC97S62> <path-to-output>\n";
  print STDERR "e.g. $0 /export/corpora3/LDC/LDC97S62 data/swbd1\n";
  exit(1);
}
($db_base, $out_dir) = @ARGV;

if (system("mkdir -p $out_dir")) {
  die "Error making directory $out_dir";
}

open(GNDR, ">$out_dir/spk2gender") || die "Could not open the output file $out_dir/spk2gender";
open(SPKR, ">$out_dir/utt2spk") || die "Could not open the output file $out_dir/utt2spk";
open(WAV, ">$out_dir/wav.scp") || die "Could not open the output file $out_dir/wav.scp";

if (-f "$db_base/caller_tab.csv") {
  open(CT, '<', "$db_base/caller_tab.csv") || die "Could not open $db_base/caller_tab.csv";
} else {
  print "File $db_base/caller_tab.csv doesn't exist, downloading it from the LDC\n";
  $url = 'https://catalog.ldc.upenn.edu/docs/LDC97S62/caller_tab.csv';
  $ct_html = qx{curl --silent $url};
  open(CT, '<', \$ct_html) || die "Could not open caller_tab.csv";
}

if (-f "$db_base/conv_tab.csv") {
  open(CS, '<', "$db_base/conv_tab.csv") || die "Could not open $db_base/conv_tab.csv";
} else {
  print "File $db_base/conv_tab.csv doesn't exist, downloading it from the LDC\n";
  $url = 'https://catalog.ldc.upenn.edu/docs/LDC97S62/conv_tab.csv';
  $cs_html = qx{curl --silent $url};
  open(CS, '<', \$cs_html) || die "Could not open conv_tab.csv";
}

$tmp_dir = "$out_base/tmp";
if (system("mkdir -p $tmp_dir") != 0) {
  die "Error making directory $tmp_dir";
}

if (system("find $db_base -name '*.sph' > $tmp_dir/sph.list") != 0) {
  die "Error getting list of sph files";
}
open(WAVLIST, "<", "$tmp_dir/sph.list") or die "cannot open wav list";
while(<WAVLIST>) {
  chomp;
  $sph = $_;
  @t = split("/",$sph);
  @t1 = split("[./]",$t[$#t]);
  $wavId=$t1[0];
  $wavs{$wavId} = $sph;
}

while(<CT>) {
  $line = $_ ;
  @A = split(", ", $line);
  $spk = "sw1_" . $A[0];
  $gender = $A[3];
  $gender =~ s/^\s+|\s+$//g;
  if ($gender eq "\"FEMALE\"") {
    print GNDR "$spk"," f\n";
  } else {
    print GNDR "$spk"," m\n";
  }
}

while (<CS>) {
  $line = $_ ;
  @A = split(", ", $line);
  $wav = "sw0" . $A[0];
  if (-e "$wavs{$wav}") {
    $spk1= "sw1_" . $A[2];
    $spk2= "sw1_" . $A[3];
    $uttId1 = $spk1 ."_" . $wav ."_1";
    print WAV "$uttId1"," sph2pipe -f wav -p -c 1 $wavs{$wav} |\n";
    print SPKR "$uttId1"," $spk1","\n";
    $uttId2 = $spk2 ."_" . $wav ."_2";
    print WAV "$uttId2"," sph2pipe -f wav -p -c 2 $wavs{$wav} |\n";
    print SPKR "$uttId2"," $spk2","\n";
  } else {
    print STDERR "Missing wav file for for $wav\n";
  }
}

close(WAV) || die;
close(SPKR) || die;
close(GNDR) || die;

if (system("utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}
if (system("utils/fix_data_dir.sh $out_dir") != 0) {
  die "Error fixing data dir $out_dir";
}
if (system("utils/validate_data_dir.sh --no-text --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}
