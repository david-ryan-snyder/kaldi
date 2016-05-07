// featbin/feat-to-len.cc

// Copyright 2016  David Snyder

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
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage = "TODO";

    ParseOptions po(usage);
    std::string extend_type = "zero";
    int32 min_feat = 100;
    po.Register("type", &extend_type, "If type=zero then we expand the "
      "feats by padding with zero on either side.  "
      "If type=copy then we pad the feats by copy the first and last "
      "frame.  If type=tile then we copy the entire feats matrix.");
    po.Register("min-feat", &min_feat, "If the number of rows is less than "
      "min-feat, then the feature matrix is expanded.");

    po.Read(argc, argv);

    if (po.NumArgs() != 1 && po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    KALDI_ASSERT(extend_type == "zero" || extend_type == "copy"
                 || extend_type == "tile");

    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);

    BaseFloatMatrixWriter writer(wspecifier);

    SequentialBaseFloatMatrixReader matrix_reader(rspecifier);
    for (; !matrix_reader.Done(); matrix_reader.Next()) {
      const Matrix<BaseFloat> &feat(matrix_reader.Value());
      Matrix<BaseFloat> new_feat(std::max(feat.NumRows(), min_feat), feat.NumCols(), kSetZero);
      int32 left_pad = (new_feat.NumRows() - feat.NumRows()) / 2;
      int32 right_pad = new_feat.NumRows() - feat.NumRows() - left_pad;
      if (feat.NumRows() < min_feat) {
        if (extend_type == "tile") {
          for (int32 i = 0; i < new_feat.NumRows(); i++) {
            new_feat.Row(i).CopyFromVec(feat.Row(i % feat.NumRows()));
          }
        } else {
          if (extend_type == "copy") {
            for (int32 i = 0; i < left_pad; i++)
              new_feat.Row(i).CopyFromVec(feat.Row(0));
            for (int32 i = 0; i < right_pad; i++)
              new_feat.Row(i + feat.NumRows() + left_pad).CopyFromVec(feat.Row(feat.NumRows() - 1));
          }
          for (int32 i = 0; i < feat.NumRows(); i++)
            new_feat.Row(i + left_pad).CopyFromVec(feat.Row(i));

        }
      } else {
        new_feat.CopyFromMat(feat);
      }
      writer.Write(matrix_reader.Key(), new_feat);
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


