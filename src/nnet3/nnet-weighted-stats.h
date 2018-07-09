// nnet3/nnet-general-component.h

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_NNET3_NNET_WEIGHTED_STATS_COMPONENT_H_
#define KALDI_NNET3_NNET_WEIGHTED_STATS_COMPONENT_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"
#include "nnet3/natural-gradient-online.h"
#include <iostream>

namespace kaldi {
namespace nnet3 {

/*
  Class WeightedStatisticsExtractionComponent is used together with
  WeightedStatisticsPoolingComponent to extract moving-average mean and
  standard-deviation statistics.

  WeightedStatisticsExtractionComponent is designed to extract statistics-- 0th-order,
  1st-order and optionally diagonal 2nd-order stats-- from small groups of
  frames, such as 10 frames.  The statistics will then be further processed by
  WeightedStatisticsPoolingComponent to compute moving-average means and (if configured)
  standard deviations.  The reason for the two-component way of doing this is
  efficiency, particularly in the graph-compilation phase.  (Otherwise there
  would be too many dependencies to process).  The WeightedStatisticsExtractionComponent
  is designed to let you extract statistics from fixed-size groups of frames
  (e.g. 10 frames), and in WeightedStatisticsPoolingComponent you are only expected to
  compute the averages at the same fixed period (e.g. 10 frames), so it's more
  efficient than if you were to compute a moving average at every single frame;
  and the computation of the intermediate stats means that most of the
  computation that goes into extracting the means and standard deviations for
  nearby frames is shared.

  The config line in a typical setup will be something like:

    input-dim=250 input-period=1 output-period=10 include-variance=true

  input-dim is self-explanatory.  The inputs will be obtained at multiples of
  input-period (e.g. it might be 3 for chain models).  output-period must be a
  multiple of input period, and the requested output indexes will be expected to
  be multiples of output-period (which you can ensure through use of the Round
  descriptor).  For instance, if you request the output on frame 80, it will
  consist of stats from input frames 80 through 89.

  An output of this component will be 'computable' any time at least one of
  the corresponding inputs is computable.

  In all cases the first dimension of the output will be a count (between 1 and
  10 inclusive in this example).  If include-variance=false, then the output
  dimension will be input-dim + 1.  and the output dimensions >0 will be
  1st-order statistics (sums of the input).  If include-variance=true, then the
  output dimension will be input-dim * 2 + 1, where the raw diagonal 2nd-order
  statistics are appended to the 0 and 1st order statistics.

  The default configuration values are:
     input-dim=-1 input-period=1 output-period=1 include-variance=true
 */
class WeightedStatisticsExtractionComponent: public Component {
 public:
  // Initializes to defaults which would not pass Check(); use InitFromConfig()
  // or Read() or copy constructor to really initialize.
  WeightedStatisticsExtractionComponent();
  // copy constructor, used in Copy().
  WeightedStatisticsExtractionComponent(const WeightedStatisticsExtractionComponent &other);

  virtual int32 InputDim() const {
    //KALDI_LOG << "input dim = " << input_dim_;
    return input_dim_;
  }
  virtual int32 OutputDim() const {
    // count + sum stats [ + sum-squared stats].
    //return input_dim_ + (include_variance_ ? input_dim_ : 0) - 1;
    //KALDI_LOG << "output dim = " << input_dim_ + (include_variance_ ? input_dim_ - 1: 0);
    return input_dim_;
  }
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "WeightedStatisticsExtractionComponent"; }
  virtual int32 Properties() const {
    return kPropagateAdds|kReordersIndexes|kBackpropNeedsInput;
  }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *, // to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.

  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Component* Copy() const {
    return new WeightedStatisticsExtractionComponent(*this);
  }

  // Some functions that are only to be reimplemented for GeneralComponents.
  virtual void GetInputIndexes(const MiscComputationInfo &misc_info,
                               const Index &output_index,
                               std::vector<Index> *desired_indexes) const;

  virtual bool IsComputable(const MiscComputationInfo &misc_info,
                            const Index &output_index,
                            const IndexSet &input_index_set,
                            std::vector<Index> *used_inputs) const;

  // This function reorders the input and output indexes so that they
  // are sorted first on n and then x and then t.
  virtual void ReorderIndexes(std::vector<Index> *input_indexes,
                              std::vector<Index> *output_indexes) const;

  virtual ComponentPrecomputedIndexes* PrecomputeIndexes(
      const MiscComputationInfo &misc_info,
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      bool need_backprop) const;

 private:
  // Checks that the parameters are valid.
  void Check() const;

  // Disallow assignment operator.
  WeightedStatisticsExtractionComponent &operator =(
      const WeightedStatisticsExtractionComponent &other);

  int32 input_dim_;
  int32 input_period_;
  int32 output_period_;
};

class WeightedStatisticsExtractionComponentPrecomputedIndexes:
      public ComponentPrecomputedIndexes {
 public:
  // While creating the output we sum over row ranges of the input.
  // forward_indexes.Dim() equals the number of rows of the output, and each
  // element is a (start, end) range of inputs, that is summed over.
  CuArray<Int32Pair> forward_indexes;

  // this vector stores the number of inputs for each output.  Normally this will be
  // the same as the component's output_period_ / input_period_, but could be less
  // due to edge effects at the utterance boundary.
  CuVector<BaseFloat> counts;

  // Each input row participates in exactly one output element, and
  // 'backward_indexes' identifies which row of the output each row
  // of the input is part of.  It's used in backprop.
  CuArray<int32> backward_indexes;

  ComponentPrecomputedIndexes *Copy() const {
    return new WeightedStatisticsExtractionComponentPrecomputedIndexes(*this);
  }

  virtual void Write(std::ostream &os, bool binary) const;

  virtual void Read(std::istream &is, bool binary);

  virtual std::string Type() const { return "WeightedStatisticsExtractionComponentPrecomputedIndexes"; }
 private:
  virtual ~WeightedStatisticsExtractionComponentPrecomputedIndexes() { }
};

/*
  Class WeightedStatisticsPoolingComponent is used together with
  WeightedStatisticsExtractionComponent to extract moving-average mean and
  standard-deviation statistics.

  WeightedStatisticsPoolingComponent pools the stats over a specified window and
  computes means and possibly log-count and stddevs from them for you.

 # In WeightedStatisticsPoolingComponent, the first element of the input is interpreted
 # as a count, which we divide by.
 # Optionally the log of the count can be output, and you can allow it to be
 # repeated several times if you want (useful for systems using the jesus-layer).
 # The output dimension is equal to num-log-count-features plus (input-dim - 1).

 # If include-log-count==false, the output dimension is the input dimension minus one.
 # If output-stddevs=true, then it expects the input-dim to be of the form 2n+1 where n is
 #  presumably the original feature dim, and it interprets the last n dimensions of the feature
 #  as a variance; it outputs the square root of the variance instead of the actual variance.

 configs and their defaults:  input-dim=-1, input-period=1, left-context=-1, right-context=-1,
    num-log-count-features=0, output-stddevs=true, variance-floor=1.0e-10

 You'd access the output of the WeightedStatisticsPoolingComponent using rounding, e.g.
  Round(component-name, 10)
 or whatever, instead of just component-name, because its output is only defined at multiples
 of its input-period.

 The output of WeightedStatisticsPoolingComponent will only be defined if at least one input was defined.
 */
class WeightedStatisticsPoolingComponent: public Component {
 public:
  // Initializes to defaults which would not pass Check(); use InitFromConfig()
  // or Read() or copy constructor to really initialize.
  WeightedStatisticsPoolingComponent();
  // copy constructor, used in Copy()
  WeightedStatisticsPoolingComponent(const WeightedStatisticsPoolingComponent &other);

  virtual int32 InputDim() const {
    //KALDI_LOG << "input dim = " << input_dim_;
    return input_dim_;
  }
  virtual int32 OutputDim() const {
    //KALDI_LOG << "output dim = " << input_dim_ + num_log_count_features_ - 1;
    return input_dim_ + num_log_count_features_ - 1;
  }
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "WeightedStatisticsPoolingComponent"; }
  virtual int32 Properties() const {
    return kReordersIndexes|kBackpropAdds|kBackpropNeedsOutput|kBackpropNeedsInput;
  }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *, // to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.

  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Component* Copy() const {
    return new WeightedStatisticsPoolingComponent(*this);
  }

  // Some functions that are only to be reimplemented for GeneralComponents.
  virtual void GetInputIndexes(const MiscComputationInfo &misc_info,
                               const Index &output_index,
                               std::vector<Index> *desired_indexes) const;

  // returns true if at least one of its inputs is computable.
  virtual bool IsComputable(const MiscComputationInfo &misc_info,
                            const Index &output_index,
                            const IndexSet &input_index_set,
                            std::vector<Index> *used_inputs) const;

  // This function reorders the input and output indexes so that they
  // are sorted first on n and then x and then t.
  virtual void ReorderIndexes(std::vector<Index> *input_indexes,
                              std::vector<Index> *output_indexes) const;

  virtual ComponentPrecomputedIndexes* PrecomputeIndexes(
      const MiscComputationInfo &misc_info,
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      bool need_backprop) const;

 private:
  // Checks that the parameters are valid.
  void Check() const;

  // Disallow assignment operator.
  WeightedStatisticsPoolingComponent &operator =(
      const WeightedStatisticsPoolingComponent &other);

  int32 input_dim_;
  int32 input_period_;
  int32 left_context_;
  int32 right_context_;
  int32 num_log_count_features_;
};

class WeightedStatisticsPoolingComponentPrecomputedIndexes:
      public ComponentPrecomputedIndexes {
 public:

  // in the first stage of creating the output we sum over row ranges of
  // the input.  forward_indexes.Dim() equals the number of rows of the
  // output, and each element is a (start, end) range of inputs, that is
  // summed over.
  CuArray<Int32Pair> forward_indexes;

  // backward_indexes contains the same information as forward_indexes, but in a
  // different format.  backward_indexes.Dim() is the same as the number of rows
  // of input, and each element contains the (start,end) of the range of outputs
  // for which this input index appears as an element of the sum for that
  // output.  This is possible because of the way the inputs and outputs are
  // ordered and because of how we select the elments to appear in the sum using
  // a window.  This quantity is used in backprop.
  CuArray<Int32Pair> backward_indexes;

  virtual ~WeightedStatisticsPoolingComponentPrecomputedIndexes() { }

  ComponentPrecomputedIndexes *Copy() const {
    return new WeightedStatisticsPoolingComponentPrecomputedIndexes(*this);
  }

  virtual void Write(std::ostream &os, bool binary) const;

  virtual void Read(std::istream &is, bool binary);

  virtual std::string Type() const { return "WeightedStatisticsPoolingComponentPrecomputedIndexes"; }
};





} // namespace nnet3
} // namespace kaldi


#endif
