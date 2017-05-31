# TODO this is a quick script for combining two PLDA scores from two
# systems.  In the future, this will be replaced by something
# better.
import sys
score1_fi = open(sys.argv[2], 'r').readlines()
trial2score = {}
for l in score1_fi:
  toks = l.rstrip().split()
  trial2score[toks[0] + toks[1]] = float(toks[2])
score2_fi = open(sys.argv[3], 'r').readlines()
for l in score2_fi:
  toks = l.rstrip().split()
  print toks[0], toks[1], (float(sys.argv[1])*trial2score[toks[0] + toks[1]] + float(toks[2]))/(1 + float(sys.argv[1]))

