// nnet3/nnet-general-component.cc

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

#include <iterator>
#include <sstream>
#include <iomanip>
#include "nnet3/nnet-general-component.h"
#include "nnet3/nnet-weighted-stats.h"
#include "nnet3/nnet-computation-graph.h"
#include "nnet3/nnet-parse.h"

namespace kaldi {
namespace nnet3 {

// used in I/O
static void CopyPairVector(const CuArray<Int32Pair> &in,
                        std::vector<std::pair<int32, int32> > *out) {
  in.CopyToVec(reinterpret_cast<std::vector<Int32Pair>*>(out));
}
// used in I/O
static void CopyPairVector(const std::vector<std::pair<int32, int32> > &in,
                        CuArray<Int32Pair> *out) {
  const std::vector<Int32Pair> *in_cast =
      reinterpret_cast<const std::vector<Int32Pair>*>(&in);
  out->CopyFromVec(*in_cast);
}

void WeightedStatisticsExtractionComponentPrecomputedIndexes::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<WeightedStatisticsExtractionComponentPrecomputedIndexes>");
  WriteToken(os, binary, "<ForwardIndexes>");
  std::vector<std::pair<int32, int32> > pairs_cpu;
  CopyPairVector(forward_indexes, &pairs_cpu);
  WriteIntegerPairVector(os, binary, pairs_cpu);
  WriteToken(os, binary, "<Counts>");
  counts.Write(os, binary);
  WriteToken(os, binary, "<BackwardIndexes>");
  std::vector<int32> backward_indexes_cpu;
  backward_indexes.CopyToVec(&backward_indexes_cpu);
  WriteIntegerVector(os, binary, backward_indexes_cpu);
  WriteToken(os, binary, "</WeightedStatisticsExtractionComponentPrecomputedIndexes>");
}

void WeightedStatisticsExtractionComponentPrecomputedIndexes::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary,
                       "<WeightedStatisticsExtractionComponentPrecomputedIndexes>",
                       "<ForwardIndexes>");
  std::vector<std::pair<int32, int32> > pairs_cpu;
  ReadIntegerPairVector(is, binary, &pairs_cpu);
  CopyPairVector(pairs_cpu, &forward_indexes);
  ExpectToken(is, binary, "<Counts>");
  counts.Read(is, binary);
  ExpectToken(is, binary, "<BackwardIndexes>");
  std::vector<int32> backward_indexes_cpu;
  ReadIntegerVector(is, binary, &backward_indexes_cpu);
  backward_indexes.CopyFromVec(backward_indexes_cpu);
  ExpectToken(is, binary, "</WeightedStatisticsExtractionComponentPrecomputedIndexes>");
}

ComponentPrecomputedIndexes*
WeightedStatisticsExtractionComponent::PrecomputeIndexes(
    const MiscComputationInfo &misc_info,
    const std::vector<Index> &input_indexes,
    const std::vector<Index> &output_indexes,
    bool need_backprop) const {
  int32 num_input_indexes = input_indexes.size(),
      num_output_indexes = output_indexes.size();
  WeightedStatisticsExtractionComponentPrecomputedIndexes *ans = new
      WeightedStatisticsExtractionComponentPrecomputedIndexes();
  // both input and output indexes are assumed sorted first on
  // n and x, then on t.
  Int32Pair invalid_pair;
  invalid_pair.first = -1;
  invalid_pair.second = -1;
  std::vector<Int32Pair> forward_indexes_cpu(output_indexes.size(),
                                             invalid_pair);
  std::vector<int32> backward_indexes_cpu(input_indexes.size(), -1);
  Vector<BaseFloat> counts_cpu(output_indexes.size());

  // this map maps from Index to the position in 'input_indexes'.
  unordered_map<Index, int32, IndexHasher> index_to_input_pos;
  for (int32 i = 0; i < num_input_indexes; i++)
    index_to_input_pos[input_indexes[i]] = i;

  for (int32 i = 0; i < num_output_indexes; i++) {
    Index output_index = output_indexes[i];
    Index input_index(output_index);
    int32 t = output_index.t,
        t_start = output_period_ * (t / output_period_);
    if (t_start > t)                // could happen for negative t_start due to
      t_start -= output_period_;    // the way modulus works in C.
    int32 t_end = t_start + output_period_;
    for (int32 t = t_start; t < t_end; t += input_period_) {
      input_index.t = t;
      unordered_map<Index, int32, IndexHasher>::iterator iter =
          index_to_input_pos.find(input_index);
      if (iter != index_to_input_pos.end()) {
        int32 input_pos = iter->second;
        if (forward_indexes_cpu[i].first == -1) {
          forward_indexes_cpu[i].first = input_pos;
          forward_indexes_cpu[i].second = input_pos + 1;
          counts_cpu(i) = 1.0;
        } else {
          // the following might fail, for instance, if the sorting
          // of the input or output indexes was not as expected.
          KALDI_ASSERT(forward_indexes_cpu[i].second == input_pos);
          forward_indexes_cpu[i].second++;
          counts_cpu(i) += 1.0;
        }
        KALDI_ASSERT(backward_indexes_cpu[input_pos] == -1);
        backward_indexes_cpu[input_pos] = i;
      }
    }
    KALDI_ASSERT(counts_cpu(i) != 0.0);
  }
  for (int32 i = 0; i < num_input_indexes; i++) {
    KALDI_ASSERT(backward_indexes_cpu[i] != -1);
  }
  ans->forward_indexes = forward_indexes_cpu;
  ans->counts = counts_cpu;
  if (need_backprop)
    ans->backward_indexes = backward_indexes_cpu;
  return ans;
}

WeightedStatisticsExtractionComponent::WeightedStatisticsExtractionComponent():
    input_dim_(-1), input_period_(1), output_period_(1) { }

WeightedStatisticsExtractionComponent::WeightedStatisticsExtractionComponent(
    const WeightedStatisticsExtractionComponent &other):
    input_dim_(other.input_dim_),
    input_period_(other.input_period_),
    output_period_(other.output_period_) {
  Check();
}

void WeightedStatisticsExtractionComponent::InitFromConfig(ConfigLine *cfl) {
  // input-dim is required.
  bool ok = cfl->GetValue("input-dim", &input_dim_);
  cfl->GetValue("input-period", &input_period_);
  cfl->GetValue("output-period", &output_period_);
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok || input_dim_ <= 0 || input_period_ <= 0 || output_period_ <= 0 ||
      (output_period_ % input_period_ != 0))
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Check();
}

void WeightedStatisticsExtractionComponent::Check() const {
  if (!(input_dim_ > 0 && input_period_ > 0 && output_period_ > 0 &&
        (output_period_ % input_period_) == 0))
    KALDI_ERR << "Invalid configuration of WeightedStatisticsExtractionComponent";
}

void WeightedStatisticsExtractionComponent::ReorderIndexes(
    std::vector<Index> *input_indexes,
    std::vector<Index> *output_indexes) const {
    std::sort(input_indexes->begin(), input_indexes->end(),
              IndexLessNxt());
    std::sort(output_indexes->begin(), output_indexes->end(),
              IndexLessNxt());
}

bool WeightedStatisticsExtractionComponent::IsComputable(
    const MiscComputationInfo &misc_info,
    const Index &output_index,
    const IndexSet &input_index_set,
    std::vector<Index> *used_inputs) const {
  Index input_index(output_index);
  int32 t = output_index.t,
      t_start = output_period_ * (t / output_period_);
  if (t_start > t)                // could happen for negative t_start due to
    t_start -= output_period_;    // the way modulus works in C.
  int32 t_end = t_start + output_period_;
  if (!used_inputs) {
    for (int32 t = t_start; t < t_end; t += input_period_) {
      input_index.t = t;
      if (input_index_set(input_index))
        return true;
    }
    return false;
  } else {
    used_inputs->clear();
    bool ans = false;
    for (int32 t = t_start; t < t_end; t += input_period_) {
      input_index.t = t;
      if (input_index_set(input_index)) {
        ans = true;
        used_inputs->push_back(input_index);
      }
    }
    return ans;
  }
}

void WeightedStatisticsExtractionComponent::GetInputIndexes(
    const MiscComputationInfo &misc_info,
    const Index &output_index,
    std::vector<Index> *desired_indexes) const {
  desired_indexes->clear();
  Index input_index(output_index);
  int32 t = output_index.t,
      t_start = output_period_ * (t / output_period_);
  if (t_start > t)                // could happen for negative t due to
    t_start -= output_period_;    // the way modulus works in C
  int32 t_end = t_start + output_period_;
  for (int32 t = t_start; t < t_end; t += input_period_) {
    input_index.t = t;
    desired_indexes->push_back(input_index);
  }
}


void* WeightedStatisticsExtractionComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes_in,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(indexes_in != NULL);
  const WeightedStatisticsExtractionComponentPrecomputedIndexes *indexes =
     dynamic_cast<const WeightedStatisticsExtractionComponentPrecomputedIndexes*>(
         indexes_in);
  int32 num_rows_out = out->NumRows();
  KALDI_ASSERT(indexes != NULL &&
               indexes->forward_indexes.Dim() == num_rows_out &&
               in.NumCols() == input_dim_ &&
               out->NumCols() == OutputDim());
  out->SetZero();

  // The first column corresponds to the counts.
  CuMatrix<BaseFloat> counts_mat(in.ColRange(0, 1));
  //counts_mat.ApplyExp();

  // This is the rest of the input, after the counts
  CuMatrix<BaseFloat> in_no_counts(in.ColRange(1, input_dim_ - 1));

  // These are the counts in Vector form
  CuVector<BaseFloat> counts(in.NumRows());
  counts.CopyColFromMat(counts_mat, 0);

  // Multiply the features by the weights
  in_no_counts.MulRowsVec(counts);

  // First element of the output is the sum of the weights
  out->ColRange(0, 1).AddRowRanges(counts_mat, indexes->forward_indexes);
  out->ColRange(1, input_dim_ - 1).AddRowRanges(in_no_counts, indexes->forward_indexes);

  return NULL;
}

void WeightedStatisticsExtractionComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes_in,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo,
    Component *, // to_update
    CuMatrixBase<BaseFloat> *in_deriv) const {

  KALDI_ASSERT(indexes_in != NULL);
  // The log-counts are in the first column of the input.
  CuMatrix<BaseFloat> counts_mat(in_value.ColRange(0, 1));
  // Now we apply exp to the log-counts to get the counts.
  //counts_mat.ApplyExp();
  // A vector version that is needed sometimes.
  CuVector<BaseFloat> counts(in_value.NumRows());
  counts.CopyColFromMat(counts_mat, 0);
  // This is the rest of the input, after the counts
  CuMatrix<BaseFloat> in_no_counts(in_value.ColRange(1, input_dim_ - 1));

  const WeightedStatisticsExtractionComponentPrecomputedIndexes *indexes =
      dynamic_cast<const WeightedStatisticsExtractionComponentPrecomputedIndexes*>(indexes_in);
  in_deriv->SetZero();

  // This is dF / dY
  in_deriv->ColRange(1, input_dim_ - 1).AddRows(1.0, out_deriv.ColRange(1, input_dim_ - 1),
                    indexes->backward_indexes);
  //in_deriv->ColRange(1, input_dim_ - 1).MulRowsVec(counts);

  // This is dF / dv
  in_deriv->ColRange(0, 1).AddRows(1.0, out_deriv.ColRange(0, 1), indexes->backward_indexes);

  // Handle the derivative for w from the weighted sum
  CuMatrix<BaseFloat> sum_deriv(in_value.NumRows(),
                                  input_dim_ - 1,
                                  kUndefined);

  // At this point, sum_deriv is dF / dY, where Y is the weighted sum
  sum_deriv.CopyRows(out_deriv.ColRange(1, input_dim_ - 1),
                         indexes->backward_indexes);

  // Now sum_deriv is (dF / dY) .* X
  sum_deriv.MulElements(in_no_counts);
  CuVector<BaseFloat> sum_deriv_vec(sum_deriv.NumRows(), kSetZero);

  // sum_deriv_vec is the sum across the cols of sum_deriv
  sum_deriv_vec.AddColSumMat(1.0, sum_deriv, 1.0);

  in_deriv->ColRange(0, 1).AddVecToCols(1.0, sum_deriv_vec, 1.0);
  //in_deriv->ColRange(0, 1).MulRowsVec(counts);
  // Now we have dF / dw = (dF / dv) * (dv / dw) + (dF / dY) * (dY / dw)
  // = (dF / dv) .* exp(w) + (dF / dY) .* X .* exp(w)

  in_deriv->ColRange(1, input_dim_ - 1).MulRowsVec(counts);
}

void WeightedStatisticsExtractionComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<WeightedStatisticsExtractionComponent>",
                       "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<InputPeriod>");
  ReadBasicType(is, binary, &input_period_);
  ExpectToken(is, binary, "<OutputPeriod>");
  ReadBasicType(is, binary, &output_period_);
  ExpectToken(is, binary, "</WeightedStatisticsExtractionComponent>");
  Check();
}

void WeightedStatisticsExtractionComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<WeightedStatisticsExtractionComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<InputPeriod>");
  WriteBasicType(os, binary, input_period_);
  WriteToken(os, binary, "<OutputPeriod>");
  WriteBasicType(os, binary, output_period_);
  WriteToken(os, binary, "</WeightedStatisticsExtractionComponent>");
}

void WeightedStatisticsPoolingComponentPrecomputedIndexes::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<WeightedStatisticsPoolingComponentPrecomputedIndexes>");
  WriteToken(os, binary, "<ForwardIndexes>");
  std::vector<std::pair<int32, int32> > indexes_cpu;
  CopyPairVector(forward_indexes, &indexes_cpu);
  WriteIntegerPairVector(os, binary, indexes_cpu);
  WriteToken(os, binary, "<BackwardIndexes>");
  CopyPairVector(backward_indexes, &indexes_cpu);
  WriteIntegerPairVector(os, binary, indexes_cpu);
  WriteToken(os, binary, "</WeightedStatisticsPoolingComponentPrecomputedIndexes>");
}

void WeightedStatisticsPoolingComponentPrecomputedIndexes::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary,
                       "<WeightedStatisticsPoolingComponentPrecomputedIndexes>",
                       "<ForwardIndexes>");
  std::vector<std::pair<int32, int32> > indexes_cpu;
  ReadIntegerPairVector(is, binary, &indexes_cpu);
  CopyPairVector(indexes_cpu, &forward_indexes);
  ExpectToken(is, binary, "<BackwardIndexes>");
  ReadIntegerPairVector(is, binary, &indexes_cpu);
  CopyPairVector(indexes_cpu, &backward_indexes);
  ExpectToken(is, binary, "</WeightedStatisticsPoolingComponentPrecomputedIndexes>");
}

void WeightedStatisticsPoolingComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = cfl->GetValue("input-dim", &input_dim_);
  cfl->GetValue("input-period", &input_period_);
  cfl->GetValue("left-context", &left_context_);
  cfl->GetValue("right-context", &right_context_);
  cfl->GetValue("num-log-count-features", &num_log_count_features_);

  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  // do some basic checks here but Check() will check more completely.
  if (!ok || input_dim_ <= 0 || left_context_ + right_context_ <= 0 ||
      num_log_count_features_ < 0)
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Check();
}

WeightedStatisticsPoolingComponent::WeightedStatisticsPoolingComponent():
    input_dim_(-1), input_period_(1), left_context_(-1), right_context_(-1),
    num_log_count_features_(0) {}


WeightedStatisticsPoolingComponent::WeightedStatisticsPoolingComponent(
    const WeightedStatisticsPoolingComponent &other):
    input_dim_(other.input_dim_), input_period_(other.input_period_),
    left_context_(other.left_context_), right_context_(other.right_context_),
    num_log_count_features_(other.num_log_count_features_) {
  Check();
}

void WeightedStatisticsPoolingComponent::Check() const {
  KALDI_ASSERT(input_dim_ > 0);
  KALDI_ASSERT(input_period_ > 0);
  KALDI_ASSERT(left_context_ >= 0 && right_context_ >= 0 &&
               left_context_ + right_context_ > 0);
  KALDI_ASSERT(left_context_ % input_period_ == 0 &&
               right_context_ % input_period_ == 0);
}

void WeightedStatisticsPoolingComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<WeightedStatisticsPoolingComponent>",
                       "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<InputPeriod>");
  ReadBasicType(is, binary, &input_period_);
  ExpectToken(is, binary, "<LeftContext>");
  ReadBasicType(is, binary, &left_context_);
  ExpectToken(is, binary, "<RightContext>");
  ReadBasicType(is, binary, &right_context_);
  ExpectToken(is, binary, "<NumLogCountFeatures>");
  ReadBasicType(is, binary, &num_log_count_features_);
  ExpectToken(is, binary, "</WeightedStatisticsPoolingComponent>");
  Check();
}

void WeightedStatisticsPoolingComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<WeightedStatisticsPoolingComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<InputPeriod>");
  WriteBasicType(os, binary, input_period_);
  WriteToken(os, binary, "<LeftContext>");
  WriteBasicType(os, binary, left_context_);
  WriteToken(os, binary, "<RightContext>");
  WriteBasicType(os, binary, right_context_);
  WriteToken(os, binary, "<NumLogCountFeatures>");
  WriteBasicType(os, binary, num_log_count_features_);
  WriteToken(os, binary, "</WeightedStatisticsPoolingComponent>");
}

void WeightedStatisticsPoolingComponent::ReorderIndexes(
    std::vector<Index> *input_indexes,
    std::vector<Index> *output_indexes) const {
    std::sort(input_indexes->begin(), input_indexes->end(),
              IndexLessNxt());
    std::sort(output_indexes->begin(), output_indexes->end(),
              IndexLessNxt());
}

void WeightedStatisticsPoolingComponent::GetInputIndexes(
    const MiscComputationInfo &misc_info,
    const Index &output_index,
    std::vector<Index> *desired_indexes) const {
  desired_indexes->clear();
  Index input_index(output_index);
  int32 middle_t = output_index.t,
      t_start = middle_t - left_context_,
      t_last = middle_t + right_context_;
  KALDI_ASSERT(middle_t % input_period_ == 0);
  for (int32 t = t_start; t <= t_last; t += input_period_) {
    input_index.t = t;
    desired_indexes->push_back(input_index);
  }
}

bool WeightedStatisticsPoolingComponent::IsComputable(
    const MiscComputationInfo &misc_info,
    const Index &output_index,
    const IndexSet &input_index_set,
    std::vector<Index> *used_inputs) const {
  if (used_inputs)
    used_inputs->clear();
  // you are not supposed to access the output of this component other than at
  // multiples of the input period.  We could make this an error but decided to
  // just have it return false.
  if (output_index.t % input_period_ != 0)
    return false;

  Index input_index(output_index);
  int32 output_t = output_index.t,
      t_start = output_t - left_context_,
      t_last = output_t + right_context_;
  if (!used_inputs) {
    for (int32 t = t_start; t <= t_last; t += input_period_) {
      input_index.t = t;
      if (input_index_set(input_index))
        return true;
    }
    return false;
  } else {
    bool ans = false;
    for (int32 t = t_start; t <= t_last; t += input_period_) {
      input_index.t = t;
      if (input_index_set(input_index)) {
        ans = true;
        used_inputs->push_back(input_index);
      }
    }
    return ans;
  }
}

ComponentPrecomputedIndexes*
WeightedStatisticsPoolingComponent::PrecomputeIndexes(
    const MiscComputationInfo &misc_info,
    const std::vector<Index> &input_indexes,
    const std::vector<Index> &output_indexes,
    bool need_backprop) const {
  int32 num_input_indexes = input_indexes.size(),
      num_output_indexes = output_indexes.size();
  WeightedStatisticsPoolingComponentPrecomputedIndexes *ans = new
      WeightedStatisticsPoolingComponentPrecomputedIndexes();

  Int32Pair invalid_pair;
  invalid_pair.first = -1;
  invalid_pair.second = -1;
  // forward_indexes_cpu[i] will be the (begin, end) of input indexes
  // included in the sum for the i'th output index.
  std::vector<Int32Pair> forward_indexes_cpu(num_output_indexes,
                                             invalid_pair);
  // backward_indexes_cpu[i] will be the (begin, end) of output indexes
  // for which the i'th input index participates in the sum.
  // because of the way the indexes are sorted (and the fact that only
  // required indexes are present at the input), it naturally has this
  // structure [i.e. no gaps in the sets of indexes].
  std::vector<Int32Pair> backward_indexes_cpu(num_input_indexes,
                                              invalid_pair);

  // this map maps from Index to the position in 'input_indexes'.
  unordered_map<Index, int32, IndexHasher> index_to_input_pos;
  for (int32 i = 0; i < num_input_indexes; i++)
    index_to_input_pos[input_indexes[i]] = i;

  for (int32 i = 0; i < num_output_indexes; i++) {
    Index input_index(output_indexes[i]);
    int32 middle_t = input_index.t,
        t_start = middle_t - left_context_,
        t_last = middle_t + right_context_;
    for (int32 t = t_start; t <= t_last; t += input_period_) {
      input_index.t = t;
      unordered_map<Index, int32, IndexHasher>::iterator iter =
          index_to_input_pos.find(input_index);
      if (iter != index_to_input_pos.end()) {
        int32 input_pos = iter->second;
        if (forward_indexes_cpu[i].first == -1) {
          forward_indexes_cpu[i].first = input_pos;
          forward_indexes_cpu[i].second = input_pos + 1;
        } else {
          KALDI_ASSERT(forward_indexes_cpu[i].second == input_pos);
          forward_indexes_cpu[i].second++;
        }
        if (backward_indexes_cpu[input_pos].first == -1) {
          backward_indexes_cpu[input_pos].first = i;
          backward_indexes_cpu[input_pos].second = i + 1;
        } else {
          KALDI_ASSERT(backward_indexes_cpu[input_pos].second == i);
          backward_indexes_cpu[input_pos].second++;
        }
      }
    }
    KALDI_ASSERT(forward_indexes_cpu[i].first != -1);
  }
  for (int32 i = 0; i < num_input_indexes; i++) {
    KALDI_ASSERT(backward_indexes_cpu[i].first != -1);
  }

  ans->forward_indexes = forward_indexes_cpu;
  if (need_backprop)
    ans->backward_indexes = backward_indexes_cpu;
  return ans;
}

void* WeightedStatisticsPoolingComponent::Propagate(
    const ComponentPrecomputedIndexes *indexes_in,
    const CuMatrixBase<BaseFloat> &in,
    CuMatrixBase<BaseFloat> *out) const {
  out->SetZero();
  KALDI_ASSERT(indexes_in != NULL);
  const WeightedStatisticsPoolingComponentPrecomputedIndexes *indexes =
      dynamic_cast<const WeightedStatisticsPoolingComponentPrecomputedIndexes*>(indexes_in);
  int32 num_rows_out = out->NumRows();
  KALDI_ASSERT(indexes != NULL &&
               indexes->forward_indexes.Dim() == num_rows_out &&
               in.NumCols() == input_dim_ &&
               out->NumCols() == OutputDim());
  CuVector<BaseFloat> counts(num_rows_out);
  // counts_mat is a fake matrix with one column, containing the counts.
  CuSubMatrix<BaseFloat> counts_mat(counts.Data(), num_rows_out, 1, 1);
  counts_mat.AddRowRanges(in.ColRange(0, 1), indexes->forward_indexes);

  CuSubMatrix<BaseFloat> out_non_count(*out, 0, num_rows_out,
                                       num_log_count_features_, input_dim_ - 1);
  out_non_count.AddRowRanges(in.ColRange(1, input_dim_ - 1),
                             indexes->forward_indexes);
  out_non_count.DivRowsVec(counts);

  if (num_log_count_features_ > 0) {
    counts.ApplyLog();
    CuVector<BaseFloat> ones(num_log_count_features_, kUndefined);
    ones.Set(1.0);
    out->ColRange(0, num_log_count_features_).AddVecVec(1.0, counts, ones);
  }
  return NULL;
}

void WeightedStatisticsPoolingComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes_in,
    const CuMatrixBase<BaseFloat> &in_value,
    const CuMatrixBase<BaseFloat> &out_value,
    const CuMatrixBase<BaseFloat> &out_deriv_in,
    void *memo,
    Component *, // to_update,
    CuMatrixBase<BaseFloat> *in_deriv) const {

  KALDI_ASSERT(indexes_in != NULL);
  const WeightedStatisticsPoolingComponentPrecomputedIndexes *indexes =
      dynamic_cast<const WeightedStatisticsPoolingComponentPrecomputedIndexes*>(
          indexes_in);
  int32 num_rows_out = out_deriv_in.NumRows();
  CuMatrix<BaseFloat> out_deriv(out_deriv_in);
  CuVector<BaseFloat> counts(num_rows_out, kUndefined);
  counts.SetZero();
  // we need to recompute the counts from the input since they are not in the
  // output.  The submatrix initializer below takes num-rows, num-cols,
  // stride;  num-cols and stride are 1.
  CuSubMatrix<BaseFloat> counts_mat(counts.Data(), num_rows_out, 1, 1);
  counts_mat.AddRowRanges(in_value.ColRange(0, 1), indexes->forward_indexes);
  // derivatives much.
  int32 feature_dim = input_dim_ - 1;

  // Now we can handle the derivative that comes from the mean
  // TODO still need to take into account log counts
  CuMatrix<BaseFloat> counts_deriv_mat(out_deriv_in.NumRows(), 1, kSetZero);
  CuVector<BaseFloat> sum_deriv_vec(out_deriv_in.NumRows(), kSetZero);
  CuMatrix<BaseFloat> out_value_out_deriv(out_value.ColRange(0, feature_dim));
  out_value_out_deriv.MulElements(out_deriv_in.ColRange(0, feature_dim));
  sum_deriv_vec.AddColSumMat(1.0, out_value_out_deriv, 1.0);
  counts_deriv_mat.AddVecToCols(1.0, sum_deriv_vec, 1.0);
  counts_deriv_mat.DivRowsVec(counts);
  counts_deriv_mat.Scale(-1.0);
  in_deriv->ColRange(0, 1).AddRowRanges(counts_deriv_mat,
    indexes->backward_indexes);
  //KALDI_LOG << "in_deriv->ColRange(0, 1) = " << in_deriv->ColRange(0, 1);

  // Divide the output derivative by the counts.  This is what we want as it
  // concerns the mean and x^2 stats.
  out_deriv.DivRowsVec(counts);
  // Now propagate the derivative back to the input.
  in_deriv->ColRange(1, input_dim_ - 1).
      AddRowRanges(out_deriv.ColRange(num_log_count_features_, input_dim_ - 1),
                   indexes->backward_indexes);
}


} // namespace nnet3
} // namespace kaldi
