#!/usr/bin/env perl
use warnings;

(scalar @ARGV == 1) or die "Usage: fix_subsegmented_feats.pl <utt2num_frames>";

my $utt2num_frames_file = $ARGV[0];

open NUM_FRAMES, $utt2num_frames_file or die "fix_subsegmented_feats.pl: Could not open file $utt2num_frames_file";

my %utt2num_frames;

while (<NUM_FRAMES>) {
  chomp;
  my @F = split;
  
  (scalar @F == 2) or die "fix_subsegmented_feats.pl: Invalid line $_ in $utt2num_frames_file";

  $utt2num_frames{$F[0]} = $F[1];
}

while (<STDIN>) {
  my $line = $_;
  
  if (m/\[([^][]*)\]\[([^][]*)\]\s*$/) {
    print ("fix_subsegmented_feats.pl: this script only supports single indices");
    exit(1);
  }
  
  my $before_range = "";
  my $range = "";

  if (m/^(.*)\[([^][]*)\]\s*$/) {
    $before_range = $1;
    $range = $2;
  } else {
    print;
    next;
  }

  my @F = split(/ /, $before_range);
  my $utt = shift @F;
  defined $utt2num_frames{$utt} or die "fix_subsegmented_feats.pl: Could not find key $utt in $utt2num_frames_file.\nError with line $line";

  if ($range !~ m/^(\d*):(\d*)([,]?.*)$/) {
    print STDERR "fix_subsegmented_feats.pl: could not make sense of input line $_";
    exit(1);
  }
    
  my $row_start = $1;
  my $row_end = $2;
  my $col_range = $3;

  if ($row_end >= $utt2num_frames{$utt}) {
    print STDERR "Fixed row_end for $utt from $row_end to $utt2num_frames{$utt}-1\n";
    $row_end = $utt2num_frames{$utt} - 1;
  }

  if ($row_start ne "") {
    $range = "$row_start:$row_end";
  } else {
    $range = "";
  }

  if ($col_range ne "") {
    $range .= ",$col_range";
  }

  print ("$utt " . join(" ", @F) . "[" . $range . "]\n");
}