// featbin/select-subvectors.cc

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
#include "matrix/kaldi-vector.h"
#include "transform/transform-common.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Select a range of elements from an archive of vectors\n"
        "\n"
        "Usage: select-subvectors [options] <vector-in-rspecifier> vector-out-wspecifier>\n"
        " e.g.: select-subvectors --offset=100 --dim=200 scp:vector.ark ark:- \n";

    bool binary = true;
    int32 offset = 0, dim = -1;
    ParseOptions po(usage);

    po.Register("binary", &binary, "Write in binary mode (only "
                "relevant if output is a wxfilename)");
    po.Register("offset", &offset,
        "The first element included in the subvector");
    po.Register("dim", &dim,
        "Total dimension of the subvector (defaults to original dim)");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string vector_in_fn = po.GetArg(1),
        vector_out_fn = po.GetArg(2);

    // all these "fn"'s are either rspecifiers or filenames.

    bool in_is_rspecifier =
        (ClassifyRspecifier(vector_in_fn, NULL, NULL)
         != kNoRspecifier),
        out_is_wspecifier =
        (ClassifyWspecifier(vector_out_fn, NULL, NULL, NULL)
         != kNoWspecifier);

    if (in_is_rspecifier != out_is_wspecifier)
      KALDI_ERR << "Cannot mix archives with regular files (copying vectors)";

    if (!in_is_rspecifier) {
      Vector<BaseFloat> vec;
      ReadKaldiObject(vector_in_fn, &vec);
      Output ko(vector_out_fn, binary);
      SubVector<BaseFloat> sub_vec(vec, offset, dim);
      Vector<BaseFloat>(sub_vec).Write(ko.Stream(), binary);
      KALDI_LOG << "Copied vector to " << vector_out_fn;
      return 0;
    } else {
      int num_done = 0;
      BaseFloatVectorWriter writer(vector_out_fn);
      SequentialBaseFloatVectorReader reader(vector_in_fn);
      for (; !reader.Done(); reader.Next(), num_done++) {
        Vector<BaseFloat> vec(reader.Value());
        SubVector<BaseFloat> sub_vec(vec, offset, dim);
        writer.Write(reader.Key(), Vector<BaseFloat>(sub_vec));
      }
      KALDI_LOG << "Copied " << num_done << " vectors";
      return (num_done != 0 ? 0 : 1);
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


