// nnet3/nnet-simple-component.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)
//                2015  Xiaohui Zhang
//                2015  Guoguo Chen
//                2015  Daniel Galvez
//                2016  Yiming Wang

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
#include <algorithm>
#include <iomanip>
#include "nnet3/nnet-power-component.h"
#include "nnet3/nnet-parse.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet3 {

void PowerComponent::Init(BaseFloat power, int32 dim) {
  power_ = power;
  dim_ = dim;
}

void PowerComponent::InitFromConfig(ConfigLine *cfl) {
  BaseFloat power = 1.0;
  int32 dim = 0;
  bool ok = cfl->GetValue("power", &power) && cfl->GetValue("dim", &dim);
  if (!ok || cfl->HasUnusedValues())
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << cfl->WholeLine() << "\"";
  Init(power, dim);
}


void* PowerComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                               const CuMatrixBase<BaseFloat> &in,
                               CuMatrixBase<BaseFloat> *out) const {
  out->CopyFromMat(in);
  //KALDI_LOG << "in.Row(0) = " << in.Row(0);
  if (power_ < 1.0 && power_ > 0.0) {
    out->ApplyFloor(1.0e-10);
  }
  out->ApplyPowAbs(power_);
  return NULL;
}

void PowerComponent::Backprop(const std::string &debug_info,
                              const ComponentPrecomputedIndexes *indexes,
                              const CuMatrixBase<BaseFloat> &in_value,
                              const CuMatrixBase<BaseFloat> &out_value,
                              const CuMatrixBase<BaseFloat> &out_deriv,
                              void *memo,
                              Component *to_update,
                              CuMatrixBase<BaseFloat> *in_deriv) const {
  if (!in_deriv)
    return;
  in_deriv->CopyFromMat(in_value);
  in_deriv->ApplyPowAbs(power_ - 1.0, true);
  in_deriv->Scale(power_);
  in_deriv->MulElements(out_deriv);
}

void PowerComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<PowerComponent>", "<Power>");
  ReadBasicType(is, binary, &power_);
  ExpectToken(is, binary, "<Dim>");
  ReadBasicType(is, binary, &dim_);
  ExpectToken(is, binary, "</PowerComponent>");
}

void PowerComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<PowerComponent>");
  WriteToken(os, binary, "<Power>");
  WriteBasicType(os, binary, power_);
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, "</PowerComponent>");
}

std::string PowerComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", dim=" << dim_
         << ", power=" << power_;
  return stream.str();
}

} // namespace nnet3
} // namespace kaldi
