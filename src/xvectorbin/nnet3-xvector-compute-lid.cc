// nnet3bin/nnet3-compute-from-egs.cc

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)

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
#include "hmm/transition-model.h"
#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-example-utils.h"
#include "nnet3/nnet-optimize.h"
#include "transform/lda-estimate.h"


namespace kaldi {
namespace nnet3 {

class NnetComputerFromEg {
 public:
  NnetComputerFromEg(const Nnet &nnet):
      nnet_(nnet), compiler_(nnet) { }

  // Compute the output (which will have the same number of rows as the number
  // of Indexes in the output of the eg), and put it in "output".
  void Compute(const NnetExample &eg, Matrix<BaseFloat> *output) {
    ComputationRequest request;
    bool need_backprop = false, store_stats = false;
    GetComputationRequest(nnet_, eg, need_backprop, store_stats, &request);
    const NnetComputation &computation = *(compiler_.Compile(request));
    NnetComputeOptions options;
    if (GetVerboseLevel() >= 3)
      options.debug = true;
    NnetComputer computer(options, computation, nnet_, NULL);
    computer.AcceptInputs(nnet_, eg.io);
    computer.Forward();
    const CuMatrixBase<BaseFloat> &nnet_output = computer.GetOutput("output");
    output->Resize(nnet_output.NumRows(), nnet_output.NumCols());
    nnet_output.CopyToMat(output);
  }
 private:
  const Nnet &nnet_;
  CachingOptimizingCompiler compiler_;

};

}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage = "TODO";

    bool compress = true;
    std::string use_gpu = "yes";
    int32 chunk_size = 500,
          min_chunk_size = -1;

    ParseOptions po(usage);
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("compress", &compress,
      "If true, compress the egs created internally.");
    po.Register("chunk-size", &chunk_size,
      "Posteriors are computed once every chunk-size and averaged");
    po.Register("min-chunk-size", &min_chunk_size,
      "Computation will fail if features are less than this.  "
      " Defaults to chunk-size.");
    if (min_chunk_size == -1)
      min_chunk_size = chunk_size;

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    std::string nnet_rxfilename = po.GetArg(1),
        feat_rspecifier = po.GetArg(2),
        vector_wspecifier = po.GetArg(3);

    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);

    NnetComputerFromEg computer(nnet);

    int32 num_success = 0,
          num_fail = 0,
          post_dim = nnet.OutputDim("output");

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    BaseFloatVectorWriter vector_writer(vector_wspecifier);
    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string utt = feat_reader.Key();
      const Matrix<BaseFloat> &feats (feat_reader.Value());
      int32 num_rows = feats.NumRows(),
            feat_dim = feats.NumCols();
      if (num_rows < min_chunk_size) {
        KALDI_WARN << "Minimum chunk size of " << min_chunk_size
                   << " is greater than the number of rows "
                   << "in utterance: " << utt << ".  Skipping.";
        num_fail++;
        continue;
      }
      int32 num_chunks = ceil(num_rows / static_cast<BaseFloat>(chunk_size));
      Vector<BaseFloat> avg(post_dim, kSetZero);
      // Iterate over the feature chunks.
      for (int32 chunk_indx = 0; chunk_indx < num_chunks; chunk_indx++) {
        // If we're nearing the end of the input, we may need to shift the
        // offset back so that we can get this_chunk_size frames of input to
        // the nnet.
        int32 offset = std::min(chunk_size, num_rows - chunk_indx * chunk_size);
        if (offset < min_chunk_size)
          continue;
        SubMatrix<BaseFloat> sub_feats(feats, chunk_indx * chunk_size, offset,
                                       0, feat_dim);

        // Create NnetExample using sub_feats.
        NnetIo nnet_input = NnetIo("input", 0, sub_feats);
        for (std::vector<Index>::iterator indx_it = nnet_input.indexes.begin();
            indx_it != nnet_input.indexes.end(); ++indx_it)
          indx_it->n = 0;
        Posterior label;
        std::vector<std::pair<int32, BaseFloat> > post;
        post.push_back(std::make_pair<int32, BaseFloat>(0, 1.0));
        label.push_back(post);
        NnetExample eg;
        eg.io.push_back(nnet_input);
        eg.io.push_back(NnetIo("output", post_dim, 0, label));
        if (compress)
          eg.Compress();
        // Done creating eg

        Matrix<BaseFloat> output;
        computer.Compute(eg, &output);
        KALDI_ASSERT(output.NumRows() != 0);
        output.ApplyExp();
        for (int32 i = 0; i < output.NumRows(); i++)
          avg.AddVec(offset, output.Row(i));
      }

      // If output is a vector, scale it by the total weight.
      avg.Scale(1.0 / avg.Sum());
      vector_writer.Write(utt, avg);
      num_success++;
    }
#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


