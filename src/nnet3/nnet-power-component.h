// nnet3/nnet-simple-component.h

// Copyright 2011-2013  Karel Vesely
//           2012-2015  Johns Hopkins University (author: Daniel Povey)
//                2013  Xiaohui Zhang
//           2014-2015  Vijayaditya Peddinti
//           2014-2015  Guoguo Chen
//                2015  Daniel Galvez
//                2015  Tom Ko

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

#ifndef KALDI_NNET3_NNET_POWER_COMPONENT_H_
#define KALDI_NNET3_NNET_POWER_COMPONENT_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"
#include "nnet3/natural-gradient-online.h"
#include <iostream>

namespace kaldi {
namespace nnet3 {

class PowerComponent: public Component {
 public:
  void Init(BaseFloat power, int32 dim);
  explicit PowerComponent(BaseFloat power, int32 dim) {
    Init(power, dim);
  }
  virtual int32 Properties() const {
    return kSimpleComponent|kBackpropNeedsInput|kBackpropNeedsOutput;
  }
  PowerComponent(): power_(1.0), dim_(0) { }
  virtual std::string Type() const { return "PowerComponent"; }
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual int32 InputDim() const { return dim_; }
  virtual int32 OutputDim() const { return dim_; }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;

  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual Component* Copy() const { return new PowerComponent(power_, dim_); }
  virtual std::string Info() const;

  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.

  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;

 protected:
  BaseFloat power_;
  int32 dim_;
};


} // namespace nnet3
} // namespace kaldi
#endif
