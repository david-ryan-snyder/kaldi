// feat/signal.h

// Copyright 2015  Tom Ko

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"

namespace kaldi {

/*
   This function implements a simple non-FFT-based convolution of two signals.
   It is suggested to use the FFT-based convolution function which is more
   efficient.
*/
void ConvolveSignals(const Vector<BaseFloat> &filter, Vector<BaseFloat> *signal);

/*
   This function implements FFT-based convolution of two signals.
   However this should be an inefficient version of BlockConvolveSignals()
   as it processes the entire signal with a single FFT.
*/
void FFTbasedConvolveSignals(const Vector<BaseFloat> &filter, Vector<BaseFloat> *signal);

/*
   This function implements FFT-based block convolution of two signals using 
   overlap-add method. This is an efficient way to evaluate the discrete 
   convolution of a long signal with a finite impulse response filter.
*/
void FFTbasedBlockConvolveSignals(const Vector<BaseFloat> &filter, Vector<BaseFloat> *signal);

}

