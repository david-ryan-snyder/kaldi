import sys, math

fi = open(sys.argv[1])
scores = []
for l in fi:
  spkr, utt, score = l.rstrip().split()
  score = float(score)
  scores.append(tuple([spkr, utt, score]))

scores = sorted(scores, key = lambda tup: tup[2], reverse=True)
out_str = ""
for x in scores:
  spkr, utt, score = x
  #out_str = out_str + spkr + ":" + str(score) + ";"
  out_str = out_str + spkr + ";"
print out_str
