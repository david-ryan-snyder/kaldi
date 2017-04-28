// ivectorbin/ivector-interpolate-plda.cc

// Copyright 2017  David Snyder

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
#include "ivector/plda.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Interpolate the parameters of two PLDA objects.\n"
        "Usage: ivector-interpolate-plda <plda-in> <plda-out>\n"
        "e.g.: ivector-interpolate-plda --factor=0.1 plda1 plda2 plda_out\n";

    ParseOptions po(usage);

    BaseFloat factor = 0.0;
    bool binary = true;
    po.Register("factor", &factor, "Factor used to interpolate the parameters"
                "of the two PLDA objects. Conceptually, this is"
                "c * plda1 + (1-c) * plda2.");
    po.Register("binary", &binary, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string plda1_rxfilename = po.GetArg(1),
        plda2_rxfilename = po.GetArg(2),
        plda_wxfilename = po.GetArg(3);

    Plda plda1, plda2;
    ReadKaldiObject(plda1_rxfilename, &plda1);
    ReadKaldiObject(plda2_rxfilename, &plda2);
    plda2.Interpolate(factor, plda1);
    WriteKaldiObject(plda2, plda_wxfilename, binary);

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
