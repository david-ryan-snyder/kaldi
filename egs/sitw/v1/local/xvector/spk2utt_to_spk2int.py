import sys

fi = open(sys.argv[1], 'r').readlines()
id = 0
for l in fi:
  t = l.rstrip().split()
  print t[0], id
  id += 1

