#!/usr/bin/env python

# This script, for use when training xvectors, decides for you which examples
# will come from which utterances, and at what point.

# You call it as (e.g.)
#
#  allocate_examples.py --min-frames-per-chunk=50 --max-frames-per-chunk=200  --frames-per-iter=1000000 \
#   --num-archives=169 --num-jobs=24  exp/xvector_a/egs/temp/utt2len.train exp/xvector_a/egs
#
# and this program outputs certain things to the temp directory (exp/xvector_a/egs/temp in this case)
# that will enable you to dump the chunks for xvector training.  What we'll eventually be doing is invoking
# the following program with something like the following args:
#
#  nnet3-xvector-get-egs [options] exp/xvector_a/temp/ranges.1  scp:data/train/feats.scp \
#    ark:exp/xvector_a/egs/egs_temp.1.ark ark:exp/xvector_a/egs/egs_temp.2.ark \
#    ark:exp/xvector_a/egs/egs_temp.3.ark
#
# where exp/xvector_a/temp/ranges.1 contains something like the following:
#
#   utt1  3  0  0   65  112  110
#   utt1  0  2  160 50  214  180
#   utt2  ...
#
# where each line is interpreted as follows:
#  <source-utterance> <relative-archive-index> <absolute-archive-index> <start-frame-index1> <num-frames1> <start-frame-index2> <num-frames2>
#
#  Note: <relative-archive-index> is the zero-based offset of the archive-index
# within the subset of archives that a particular ranges file corresponds to;
# and <absolute-archive-index> is the 1-based numeric index of the destination
# archive among the entire list of archives, which will form part of the
# archive's filename (e.g. egs/egs.<absolute-archive-index>.ark);
# <absolute-archive-index> is only kept for debug purposes so you can see which
# archive each line corresponds to.
#
# and for each line we create an eg (containing two possibly-different-length chunks of data from the
# same utterance), to one of the output archives.  The list of archives corresponding to
# ranges.n will be written to output.n, so in exp/xvector_a/temp/outputs.1 we'd have:
#
#  ark:exp/xvector_a/egs/egs_temp.1.ark ark:exp/xvector_a/egs/egs_temp.2.ark ark:exp/xvector_a/egs/egs_temp.3.ark
#
# The number of these files will equal 'num-jobs'.  If you add up the word-counts of
# all the outputs.* files you'll get 'num-archives'.  The number of frames in each archive
# will be about the --frames-per-iter.
#
# This program will also output to the temp directory a file called archive_chunk_lengths which gives you
# the pairs of frame-lengths associated with each archives. e.g.
# 1   60  180
# 2   120  85
# the format is:  <archive-index> <num-frames1> <num-frames2>.
# the <num-frames1> and <num-frames2> will always be in the range
# [min-frames-per-chunk, max-frames-per-chunk].



# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings, random


parser = argparse.ArgumentParser(description="Writes ranges.*, outputs.* and archive_chunk_lengths files "
                                 "in preparation for dumping egs for xvector training.",
                                 epilog="Called by steps/nnet3/xvector/get_egs.sh")
parser.add_argument("--prefix", type=str, default="",
                   help="Adds a prefix to the output files. This is used to distinguish between the train "
                   "and diagnostic files.")
parser.add_argument("--min-frames-per-chunk", type=int, default=50,
                    help="Minimum number of frames-per-chunk used for any archive")
parser.add_argument("--max-frames-per-chunk", type=int, default=300,
                    help="Maximum number of frames-per-chunk used for any archive")
parser.add_argument("--randomize-chunk-length", type=str,
                    help="If true, randomly pick a chunk length in [min-frames-per-chunk, max-frames-per-chunk]."
                    "If false, the chunk length varies from min-frames-per-chunk to max-frames-per-chunk"
                    "according to a geometric sequence.",
                    default="true", choices = ["false", "true"])
parser.add_argument("--frames-per-iter", type=int, default=1000000,
                    help="Target number of frames for each archive")
parser.add_argument("--num-archives", type=int, default=-1,
                    help="Number of archives to write");
parser.add_argument("--num-jobs", type=int, default=-1,
                    help="Number of jobs we're going to use to write the archives; the ranges.* "
                    "and outputs.* files are indexed by job.  Must be <= the --num-archives option.");
parser.add_argument("--seed", type=int, default=1,
                    help="Seed for random number generator")
parser.add_argument("--num-pdfs", type=int, default=-1,
                    help="Num pdfs")

# now the positional arguments
parser.add_argument("utt2len",
                    help="utt2len file of the features to be used as input (format is: "
                    "<utterance-id> <approx-num-frames>)");
parser.add_argument("utt2lang",
                    help="utt2lang file of the features to be used as input (format is: "
                    "<utterance-id> <lang-id>)");
parser.add_argument("egs_dir",
                    help="Name of egs directory, e.g. exp/xvector_a/egs");

print(' '.join(sys.argv))

args = parser.parse_args()

if not os.path.exists(args.egs_dir + "/temp"):
    os.makedirs(args.egs_dir + "/temp")

## Check arguments.
if args.min_frames_per_chunk <= 1:
    sys.exit("--min-frames-per-chunk is invalid.")
if args.max_frames_per_chunk < args.min_frames_per_chunk:
    sys.exit("--max-frames-per-chunk is invalid.")
if args.frames_per_iter < 1000:
    sys.exit("--frames-per-iter is invalid.")
if args.num_archives < 1:
    sys.exit("--num-archives is invalid")
if args.num_jobs > args.num_archives:
    sys.exit("--num-jobs is invalid (must not exceed num-archives)")


random.seed(args.seed)


f = open(args.utt2len, "r");
if f is None:
    sys.exit("Error opening utt2len file " + str(args.utt2len));
utt_ids = []
lengths = []
for line in f:
    a = line.split()
    if len(a) != 2:
        sys.exit("bad line in utt2len file " + line);
    utt_ids.append(a[0])
    lengths.append(int(a[1]))
f.close()

f = open(args.utt2lang, "r");
if f is None:
    sys.exit("Error opening utt2lang file " + str(args.utt2lang));
langs = []
for line in f:
    a = line.split()
    if len(a) != 2 or utt_ids[len(langs)] != a[0]:
        sys.exit("bad line in utt2lang file " + line);
    langs.append(int(a[1]))
f.close()

lang_list = list(set(langs))

if args.num_pdfs == -1:
  args.num_pdfs = max(langs) + 1

num_utts = len(utt_ids)
max_length = max(lengths)

# this function returns a random integer utterance index, limited to utterances
# above a minimum length in frames, with probability proportional to its length.
def RandomUttAtLeastThisLong(min_length):
    lang_id = random.randrange(0, len(lang_list))
    lang = lang_list[lang_id]
    while True:
        i = random.randrange(0, num_utts)
        # read the next line as 'with probability lengths[i] / max_length'.
        # this allows us to draw utterances with probability with
        # prob proportional to their length.
        if langs[i] == lang and lengths[i] > min_length and random.random() < lengths[i] / float(max_length):
            return i

def RandomChunkLength():
    ans = random.randint(args.min_frames_per_chunk, args.max_frames_per_chunk)
    return ans

# This function returns an integer in the range
# [min-frames-per-chunk, max-frames-per-chunk] according to a geometric
# sequence. For example, suppose min-frames-per-chunk is 50,
# max-frames-per-chunk is 200, and args.num_archives is 3. Then the
# lengths for archives 0, 1, and 2 will be 50, 100, and 200.
def DeterministicChunkLength(archive_id):
  if args.max_frames_per_chunk == args.min_frames_per_chunk:
    return args.max_frames_per_chunk
  else:
    return int(math.pow(float(args.max_frames_per_chunk) /
                     args.min_frames_per_chunk, float(archive_id) /
                     (args.num_archives-1)) * args.min_frames_per_chunk + 0.5)



# given an utterance length utt_length (in frames) and two desired chunk lengths
# (length1 and length2) whose sum is <= utt_length,
# this function randomly picks the starting points of the chunks for you.
# the chunks may appear randomly in either order.
def GetRandomOffset(utt_length, length):
    if length > utt_length:
        sys.exit("code error: length > utt-length")
    free_length = utt_length - length

    offset = random.randint(0, free_length)
    return offset


# archive_chunk_lengths and all_archives will be arrays of dimension
# args.num_archives.  archive_chunk_lengths contains 2-tuples
# (left-num-frames, right-num-frames).
archive_chunk_lengths = []  # archive
# each element of all_egs (one per archive) is
# an array of 3-tuples (utterance-index, offset1, offset2)
all_egs= []

prefix = ""
if args.prefix != "":
  prefix = args.prefix + "_"

info_f = open(args.egs_dir + "/temp/" + prefix + "archive_chunk_lengths", "w")
if info_f is None:
    sys.exit(str("Error opening file {0}/temp/" + prefix + "archive_chunk_lengths").format(args.egs_dir));
for archive_index in range(args.num_archives):
    print("Processing archive {0}".format(archive_index + 1))
    if args.randomize_chunk_length == "true":
        # don't constrain the lengths to be the same
        length = RandomChunkLength();
    else:
        length = DeterministicChunkLength(archive_index);
    print("{0} {1}".format(archive_index + 1, length), file=info_f)
    archive_chunk_lengths.append(length)
    this_num_egs = (args.frames_per_iter / length) + 1
    this_egs = [ ] # this will be an array of 3-tuples (utterance-index, left-start-frame, right-start-frame).
    for n in range(this_num_egs):
        utt_index = RandomUttAtLeastThisLong(length)
        utt_len = lengths[utt_index]
        offset = GetRandomOffset(utt_len, length)
        this_egs.append( (utt_index, offset) )
    all_egs.append(this_egs)
info_f.close()

# work out how many archives we assign to each job in an equitable way.
num_archives_per_job = [ 0 ] * args.num_jobs
for i in range(0, args.num_archives):
    num_archives_per_job[i % args.num_jobs]  = num_archives_per_job[i % args.num_jobs] + 1

pdf2num = {}
cur_archive = 0
for job in range(args.num_jobs):
    this_ranges = []
    this_archives_for_job = []
    this_num_archives = num_archives_per_job[job]

    for i in range(0, this_num_archives):
        this_archives_for_job.append(cur_archive)
        for (utterance_index, offset) in all_egs[cur_archive]:
            this_ranges.append( (utterance_index, i, offset) )
        cur_archive = cur_archive + 1

    f = open(args.egs_dir + "/temp/" + prefix + "ranges." + str(job + 1), "w")
    if f is None:
        sys.exit("Error opening file " + args.egs_dir + "/temp/" + prefix + "ranges." + str(job + 1))
    for (utterance_index, i, offset) in sorted(this_ranges):
        archive_index = this_archives_for_job[i]
        print("{0} {1} {2} {3} {4} {5}".format(utt_ids[utterance_index],
                                           i,
                                           archive_index + 1,
                                           offset,
                                           archive_chunk_lengths[archive_index],
                                           langs[utterance_index]),
              file=f)
        if langs[utterance_index] in pdf2num:
          pdf2num[langs[utterance_index]] += 1
        else:
          pdf2num[langs[utterance_index]] = 1
    f.close()


    f = open(args.egs_dir + "/temp/" + prefix + "outputs." + str(job + 1), "w")
    if f is None:
        sys.exit("Error opening file " + args.egs_dir + "/temp/" + prefix + "outputs." + str(job + 1))
    print( " ".join([ str("{0}/" + prefix + "egs_temp.{1}.ark").format(args.egs_dir, n + 1) for n in this_archives_for_job ]),
           file=f)
    f.close()

f = open(args.egs_dir + "/" + prefix + "pdf2num", "w")
nums = []
for k in range(0, args.num_pdfs):
  if k in pdf2num:
    nums.append(pdf2num[k])
  else:
    nums.append(0)

print(" ".join(map(str, nums)), file=f)
f.close()

print("allocate_examples_lid.py: finished generating " + prefix + "ranges.* and " + prefix + "outputs.* files")

