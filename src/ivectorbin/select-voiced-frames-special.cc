// ivectorbin/select-voiced-frames-special.cc

// Copyright   2013   Daniel Povey
//             2017   David Snyder

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
#include "feat/feature-functions.h"


namespace kaldi {
// Check if the frame should be kept.  This happens if the frame at index i
// is speech, or if there is a speech frame in the window [i - left-context,
// i + right_context].
bool KeepFrame(const Vector<BaseFloat> &voiced, int32 i, int32 left_context,
    int32 right_context) {

  if (voiced(i) == 1.0) {
    return true;

  } else {
    KALDI_ASSERT(voiced(i) == 0.0);
    for (int32 j = i - 1; j >= std::max(0, i - left_context); j--) {
      if (voiced(j) == 1.0) {
        return true;
      }
    }

    for (int32 j = i + 1; j <= std::min(voiced.Dim(), i + right_context);
        j++) {
      if (voiced(j) == 1.0) {
        return true;
      }
    }
  }
  return false;
}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "TODO: This binary is a work in progress.  This functionality might\n"
        "get added to the binary select-voiced-frames instead of being a\n"
        "standalone program.\n"
        "\n"
        "This binary is similar to select-voiced-frames, but provides\n"
        "options to enable keeping some silence frames around speech frames\n"
        "using --left-context and --right-context.\n"
        "Usage: select-voiced-frames-special [options] <feats-rspecifier> "
        " <vad-rspecifier> <feats-wspecifier>\n"
        "E.g.: select-voiced-frames [options] scp:feats.scp scp:vad.scp ark:-\n";

    ParseOptions po(usage);

    int32 left_context = 2,
          right_context = 2;

    po.Register("left-context", &left_context, "Keep silence frame if"
        " there's a speech frame within this many frames to the left.");
    po.Register("right-context", &right_context, "Keep silence frame if"
        " there's a speech frame within this many frames to the right.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string feat_rspecifier = po.GetArg(1),
        vad_rspecifier = po.GetArg(2),
        feat_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    RandomAccessBaseFloatVectorReader vad_reader(vad_rspecifier);
    BaseFloatMatrixWriter feat_writer(feat_wspecifier);

    int32 num_done = 0, num_err = 0;

    for (;!feat_reader.Done(); feat_reader.Next()) {
      std::string utt = feat_reader.Key();
      const Matrix<BaseFloat> &feat = feat_reader.Value();
      if (feat.NumRows() == 0) {
        KALDI_WARN << "Empty feature matrix for utterance " << utt;
        num_err++;
        continue;
      }
      if (!vad_reader.HasKey(utt)) {
        KALDI_WARN << "No VAD input found for utterance " << utt;
        num_err++;
        continue;
      }
      const Vector<BaseFloat> &voiced = vad_reader.Value(utt);

      if (feat.NumRows() != voiced.Dim()) {
        KALDI_WARN << "Mismatch in number for frames " << feat.NumRows()
                   << " for features and VAD " << voiced.Dim()
                   << ", for utterance " << utt;
        num_err++;
        continue;
      }
      if (voiced.Sum() == 0.0) {
        KALDI_WARN << "No features were judged as voiced for utterance "
                   << utt;
        num_err++;
        continue;
      }

      std::vector<int32> frames_to_keep;
      for (int32 i = 0; i < feat.NumRows(); i++) {
        if (KeepFrame(voiced, i, left_context, right_context))
          frames_to_keep.push_back(i);
      }
      Matrix<BaseFloat> voiced_feat(frames_to_keep.size(), feat.NumCols());
      for (int32 i = 0; i < frames_to_keep.size(); i++) {
          voiced_feat.Row(i).CopyFromVec(feat.Row(frames_to_keep[i]));
      }
      feat_writer.Write(utt, voiced_feat);
      num_done++;
    }

    KALDI_LOG << "Done selecting voiced frames; processed "
              << num_done << " utterances, "
              << num_err << " had errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


