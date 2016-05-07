#!/usr/bin/env python

# This script is for making the neural-net configs for the x-vector training
# for a speaker-id type setup (where it has to classify short chunks of frames,
# e.g. 1 or 2 seconds long, as being from the same original utterance).
# It's mostly the same as ../make_jesus_configs.py.

# The neural net just takes in a sequence of frames and outputs a vector.  It
# takes an input named 'input' which will typically be the MFCCs; and it
# produces an output named 'output' which during training will only be accessed
# on t=0; and which is the vector representation of the entire input.
# The config file itself has the same structure as for a regular TDNN, for
# the most part, except for a few simplifications and tweaks (such the
# --output-scale option).
#
# Note, the absolute time values actually matter because we do the pooling using
# the  StatisticsExtractionComponent and StatisticsPoolingComponent, and we have
# to specify what range of time-offsets we pool over.  We'll let the user
# specify the time begin and end values of the features in the egs to match
# whatever configuration is specified to here (e.g. if the statistics configuration
# is -99:3:9:0, then the user would specify something like -99 to 0 as the time
# range to use in the egs.

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings


parser = argparse.ArgumentParser(description="Writes config files and variables "
                                 "for TDNNs with jesus-layer nonlinearity for use in "
                                 "training xvectors.",
                                 epilog="Typically called prior to steps/nnet3/xvector train_tdnn.sh; "
                                 "see egs/swbd/s5c/local/xvector/run.sh for example.");
parser.add_argument("--splice-indexes", type=str,
                    help="Splice[:recurrence] indexes at each hidden layer, e.g. '-3,-2,-1,0,1,2,3 -3,0:-3 -3,0:-3 -6,-3,0:-6,-3'. "
                    "Note: recurrence indexes are optional, may not appear in 1st layer, and must be "
                    "either all negative or all positive for any given layer.")
parser.add_argument("--feat-dim", type=int,
                    help="Raw feature dimension, e.g. 13")
parser.add_argument("--self-repair-scale", type=float,
                    help="Small scale involved in preventing inputs to nonlinearities getting out of range",
                    default=0.00002)
parser.add_argument("--jesus-hidden-dim", type=int,
                    help="hidden dimension of Jesus layer.", default=3500)
parser.add_argument("--jesus-output-dim", type=int,
                    help="part of output dimension of Jesus layer that goes to next layer",
                    default=1000)
parser.add_argument("--jesus-input-dim", type=int,
                    help="Input dimension of Jesus layer that comes from affine projection "
                    "from the previous layer (same as output dim of forward affine transform)",
                    default=1000)
parser.add_argument("--final-hidden-dim", type=int,
                    help="Final hidden layer dimension-- or if <0, the same as "
                    "--jesus-input-dim", default=-1)
parser.add_argument("--num-jesus-blocks", type=int,
                    help="number of blocks in Jesus layer.  All configs of the form "
                    "--jesus-*-dim will be rounded up to be a multiple of this.",
                    default=100);
parser.add_argument("--jesus-stddev-scale", type=float,
                    help="Scaling factor on parameter stddev of Jesus layer (smaller->jesus layer learns faster)",
                    default=1.0)
parser.add_argument("prior_scales",
                    help="Filename for the prior scales file");
parser.add_argument("config_out",
                    help="Filename for the config file to be written to");
parser.add_argument("--num-targets", type=int,
                       help="number of network targets (e.g. num-pdf-ids")

print(' '.join(sys.argv))

args = parser.parse_args()

## Check arguments.
if args.splice_indexes is None:
    sys.exit("--splice-indexes argument is required");
if args.feat_dim is None or not (args.feat_dim > 0):
    sys.exit("--feat-dim argument is required");
if args.num_jesus_blocks < 1:
    sys.exit("invalid --num-jesus-blocks value");
if args.final_hidden_dim < 0:
    args.final_hidden_dim = args.jesus_input_dim
if not args.num_targets > 0:
    print(args.num_targets)
    raise Exception("num_targets has to be positive")

for name in [ "jesus_hidden_dim", "jesus_output_dim", "jesus_input_dim",
              "final_hidden_dim" ]:
    old_val = getattr(args, name)
    if old_val % args.num_jesus_blocks != 0:
        new_val = old_val + args.num_jesus_blocks - (old_val % args.num_jesus_blocks)
        printable_name = '--' + name.replace('_', '-')
        print('Rounding up {0} from {1} to {2} to be a multiple of --num-jesus-blocks={3}: '.format(
                printable_name, old_val, new_val, args.num_jesus_blocks))
        setattr(args, name, new_val);

# this is a bit like a struct, initialized from a string, which describes how to
# set up the statistics-pooling and statistics-extraction components.
# An example string is 'mean(-99:3:9::99)', which means, compute the mean of
# data within a window of -99 to +99, with distinct means computed every 9 frames
# (we round to get the appropriate one), and with the input extracted on multiples
# of 3 frames (so this will force the input to this layer to be evaluated
# every 3 frames).  Another example string is 'mean+stddev(-99:3:9:99)',
# which will also cause the standard deviation to be computed; or
# 'mean+stddev+count(-99:3:9:99)'.
class StatisticsConfig:
    # e.g. c = StatisticsConfig('mean+stddev(-99:3:9:99)', 400, 100, 'jesus1-output-affine')
    # e.g. c = StatisticsConfig('mean+stddev+count(-99:3:9:99)', 400, 100, 'jesus1-output-affine')
    def __init__(self, config_string, input_dim, num_jesus_blocks, input_name):
        self.input_dim = input_dim
        self.num_jesus_blocks = num_jesus_blocks  # we need to know this because
                                                  # it's the dimension of the count
                                                  # features that we output.
        self.input_name = input_name

        m = re.search("mean(|\+stddev)(|\+count)\((-?\d+):(-?\d+):(-?\d+):(-?\d+)\)",
                      config_string)
        if m == None:
            sys.exit("Invalid splice-index or statistics-config string: " + config_string)
        self.output_stddev = (m.group(1) == '+stddev')
        self.output_count = (m.group(2) == '+count')

        self.left_context = -int(m.group(3))
        self.input_period = int(m.group(4))
        self.stats_period = int(m.group(5))
        self.right_context = int(m.group(6))
        if not (self.left_context >= 0 and self.right_context >= 0 and
                self.input_period > 0 and self.stats_period > 0 and
                self.num_jesus_blocks > 0 and
                self.left_context % self.stats_period == 0 and
                self.right_context % self.stats_period == 0 and
                self.stats_period % self.input_period == 0):
            sys.exit("Invalid configuration of statistics-extraction: " + config_string)

    # OutputDim() returns the output dimension of the node that this produces.
    def OutputDim(self):
        return self.input_dim * (2 if self.output_stddev else 1) + (self.num_jesus_blocks if self.output_count else 0)

    # OutputDims() returns an array of output dimensions... this node produces
    # one output node, but this array explains how it's split up into different types
    # of output (which will affect how we reorder the indexes for the jesus-layer).
    def OutputDims(self):
        ans = [ self.input_dim ]
        if self.output_stddev:
            ans.append(self.input_dim)
        if self.output_count:
            ans.append(self.num_jesus_blocks)
        return ans

    # Descriptor() returns the textual form of the descriptor by which the
    # output of this node is to be accessed.
    def Descriptor(self):
        return 'Round({0}-pooling-{1}-{2}, {3})'.format(self.input_name, self.left_context,
                                                        self.right_context, self.stats_period)

    # This function writes the configuration lines need to compute the specified
    # statistics, to the file f.
    def WriteConfigs(self, f):
        print('component name={0}-extraction-{1}-{2} type=StatisticsExtractionComponent input-dim={3} '
              'input-period={4} output-period={5} include-variance={6} '.format(
                self.input_name, self.left_context, self.right_context,
                self.input_dim, self.input_period, self.stats_period,
                ('true' if self.output_stddev else 'false')), file=f)
        print('component-node name={0}-extraction-{1}-{2} component={0}-extraction-{1}-{2} input={0} '.format(
                self.input_name, self.left_context, self.right_context), file=f)
        stats_dim = 1 + self.input_dim * (2 if self.output_stddev else 1)
        print('component name={0}-pooling-{1}-{2} type=StatisticsPoolingComponent input-dim={3} '
              'input-period={4} left-context={1} right-context={2} num-log-count-features={6} '
              'output-stddevs={5} '.format(self.input_name, self.left_context, self.right_context,
                                           stats_dim, self.stats_period,
                                           ('true' if self.output_stddev else 'false'),
                                           (self.num_jesus_blocks if self.output_count else 0)),
              file=f)
        print('component-node name={0}-pooling-{1}-{2} component={0}-pooling-{1}-{2} input={0}-extraction-{1}-{2} '.format(
                self.input_name, self.left_context, self.right_context), file=f)




## Work out splice_array
## e.g. for
## args.splice_indexes == '-3,-2,-1,0,1,2,3 -3,0:-3 -3,0:-3 -6,-3,0:-6,-3'
## we would have
##   splice_array = [ [ -3,-2,...3 ], [-3,0] [-3,0] [-6,-3,0]


splice_array = []
left_context = 0
right_context = 0
split_on_spaces = args.splice_indexes.split(" ");  # we already checked the string is nonempty.
if len(split_on_spaces) < 2:
    sys.exit("invalid --splice-indexes argument, too short: "
             + args.splice_indexes)
try:
    for string in split_on_spaces:
        this_layer = len(splice_array)

        this_splices = string.split(",")
        splice_array.append(this_splices)
        # the rest of this block just does some checking.
        for s in this_splices:
            try:
                n = int(s)
            except:
                if len(splice_array) == 1:
                    sys.exit("First dimension of splicing array must not have averaging [yet]")
                #try:
                x = StatisticsConfig(s, 100, 100, 'foo')
                # don't catch the exception, let it make the program die.
                #except:
                #    sys.exit("The following element of the splicing array is not a valid specifier "
                #    "of statistics: " + s)

except ValueError as e:
    sys.exit("invalid --splice-indexes argument " + args.splice_indexes + " " + str(e))

num_hidden_layers = len(splice_array)


# we just write a single config file.
f = open(args.config_out, "w")

print('input-node name=input dim=' + str(args.feat_dim), file=f)
cur_output = 'input'
cur_affine_output_dim = args.feat_dim

for l in range(1, num_hidden_layers + 1):
    # the following summarizes the structure of the layers:  Here, the Jesus component includes ReLU at its input and output, and renormalize
    #   at its output after the ReLU.
    # layer1: splice + affine + ReLU + renormalize
    # layerX: splice + Jesus + affine + ReLU

    # Inside the jesus component is:
    #  [permute +] ReLU + repeated-affine + ReLU + repeated-affine
    # [we make the repeated-affine the last one so we don't have to redo that in backprop].
    # We follow this with a post-jesus composite component containing the operations:
    #  [permute +] ReLU + renormalize
    # call this post-jesusN.
    # After this we use dim-range nodes to split up the output into
    # [ jesusN-output, jesusN-direct-output and jesusN-projected-output ]
    # parts;
    # and nodes for the jesusN-affine.

    print('# Config file for layer {0} of the network'.format(l), file=f)

    splices = []
    spliced_dims = []
    for s in splice_array[l-1]:
        # the connection from the previous layer
        try:
            offset = int(s)
            # it's an integer offset.
            splices.append('Offset({0}, {1})'.format(cur_output, offset))
            spliced_dims.append(cur_affine_output_dim)
        except:
            # it's not an integer offset, so assume it specifies the
            # statistics-extraction.
            stats = StatisticsConfig(s, cur_affine_output_dim,
                                     args.num_jesus_blocks, cur_output)
            stats.WriteConfigs(f)
            splices.append(stats.Descriptor())
            spliced_dims.extend(stats.OutputDims())

    # get the input to the Jesus layer.
    cur_input = 'Append({0})'.format(', '.join(splices))
    cur_dim = sum(spliced_dims)

    if l == 1:
        # just have an affine component for the first hidden layer.
        # we don't need a nonlinearity as there is one at the input of
        # the jesus component.
        print('component name=x-affine1 type=NaturalGradientAffineComponent '
              'input-dim={0} output-dim={1} bias-stddev=0'.format(
                cur_dim, args.jesus_input_dim), file=f)
        print('component-node name=x-affine1 component=x-affine1 input={0}'.format(
                cur_input), file=f)
        cur_affine_output_dim = args.jesus_input_dim
        cur_output = 'x-affine1'
        continue


    # As input to the Jesus component we'll append the spliced input and any
    # mean/stddev-stats input, and the first thing inside the component that
    # we do is rearrange the dimensions so that things pertaining to a
    # particular block stay together.

    column_map = []
    for x in range(0, args.num_jesus_blocks):
        dim_offset = 0
        for src_splice in spliced_dims:
            src_block_size = src_splice / args.num_jesus_blocks
            for y in range(0, src_block_size):
                column_map.append(dim_offset + (x * src_block_size) + y)
            dim_offset += src_splice
    if sorted(column_map) != range(0, sum(spliced_dims)):
        print("column_map is " + str(column_map))
        print("num_jesus_blocks is " + str(args.num_jesus_blocks))
        print("spliced_dims is " + str(spliced_dims))
        sys.exit("code error creating new column order")

    need_input_permute_component = (column_map != range(0, sum(spliced_dims)))

    # Now add the jesus component.
    num_sub_components = (5 if need_input_permute_component else 4);
    print('component name=x-jesus{0} type=CompositeComponent num-components={1}'.format(
            l, num_sub_components), file=f, end='')
    # print the sub-components of the CompositeComopnent on the same line.
    # this CompositeComponent has the same effect as a sequence of
    # components, but saves memory.
    if need_input_permute_component:
        print(" component1='type=PermuteComponent column-map={1}'".format(
                l, ','.join([str(x) for x in column_map])), file=f, end='')
    print(" component{0}='type=RectifiedLinearComponent dim={1} self-repair-scale={2}'".format(
            (2 if need_input_permute_component else 1),
            cur_dim, args.self_repair_scale), file=f, end='')

    print(" component{0}='type=NaturalGradientRepeatedAffineComponent input-dim={1} output-dim={2} "
          "num-repeats={3} param-stddev={4} bias-mean={5} bias-stddev=0'".format(
            (3 if need_input_permute_component else 2),
            cur_dim, args.jesus_hidden_dim,
            args.num_jesus_blocks,
            args.jesus_stddev_scale / math.sqrt(cur_dim / args.num_jesus_blocks),
            0.5 * args.jesus_stddev_scale),
          file=f, end='')

    print(" component{0}='type=RectifiedLinearComponent dim={1} self-repair-scale={2}'".format(
            (4 if need_input_permute_component else 3),
            args.jesus_hidden_dim, args.self_repair_scale), file=f, end='')

    print(" component{0}='type=NaturalGradientRepeatedAffineComponent input-dim={1} output-dim={2} "
          "num-repeats={3} param-stddev={4} bias-mean={5} bias-stddev=0'".format(
            (5 if need_input_permute_component else 4),
            args.jesus_hidden_dim,
            args.jesus_output_dim,
            args.num_jesus_blocks,
            args.jesus_stddev_scale / math.sqrt(args.jesus_hidden_dim / args.num_jesus_blocks),
            0.5 * args.jesus_stddev_scale),
          file=f, end='')

    print("", file=f) # print newline.
    print('component-node name=x-jesus{0} component=x-jesus{0} input={1}'.format(
            l, cur_input), file=f)

    # now print the post-Jesus component which consists of ReLU +
    # renormalize.

    num_sub_components = 2
    print('component name=x-post-jesus{0} type=CompositeComponent num-components=2'.format(l),
          file=f, end='')

    # still within the post-Jesus component, print the ReLU
    print(" component1='type=RectifiedLinearComponent dim={0} self-repair-scale={1}'".format(
            args.jesus_output_dim, args.self_repair_scale), file=f, end='')
    # still within the post-Jesus component, print the NormalizeComponent
    print(" component2='type=NormalizeComponent dim={0} '".format(
            args.jesus_output_dim), file=f, end='')
    print("", file=f) # print newline.
    print('component-node name=x-post-jesus{0} component=x-post-jesus{0} input=x-jesus{0}'.format(l),
          file=f)

    cur_affine_output_dim = (args.jesus_input_dim if l < num_hidden_layers else args.final_hidden_dim)
    print('component name=x-affine{0} type=NaturalGradientAffineComponent '
          'input-dim={1} output-dim={2} bias-stddev=0'.
          format(l, args.jesus_output_dim, cur_affine_output_dim), file=f)
    print('component-node name=x-jesus{0}-output-affine component=x-affine{0} input=x-post-jesus{0}'.format(
            l), file=f)
    cur_output = 'x-jesus{0}-output-affine'.format(l)

print('component name=x-final-relu type=RectifiedLinearComponent dim={0} self-repair-scale={1}'.format(
        cur_affine_output_dim, args.self_repair_scale), file=f)
print('component-node name=x-final-relu component=x-final-relu input={0}'.format(cur_output),
      file=f)
print('component name=x-final-affine type=NaturalGradientAffineComponent '
      'input-dim={0} output-dim={1} bias-stddev=0'.format(
        cur_affine_output_dim, args.num_targets), file=f)
print('component-node name=x-final-affine component=x-final-affine input=x-final-relu',
      file=f)
print('component name=final-fixed-scale type=FixedScaleComponent '
                  'scales={0}'.format(
                    args.prior_scales), file=f)
print('component-node name=final-fixed-scale component=final-fixed-scale input=x-final-affine',
      file=f)
print('component name=final-log-softmax type=LogSoftmaxComponent dim={0}'.format(
                args.num_targets), file=f)
print('component-node name=final-log-softmax component=final-log-softmax '
      'input=final-fixed-scale', file=f)
print('output-node name=output input=final-log-softmax', file=f)
f.close()

