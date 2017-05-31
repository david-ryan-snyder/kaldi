// nnet3bin/nnet3-compute.cc

// Copyright 2012-2015   Johns Hopkins University (author: Daniel Povey)
//                2016   David Snyder

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
#include "base/timer.h"
#include "nnet3/nnet-utils.h"
#include "xvector/nnet-xvector-compute.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-example-utils.h"

namespace kaldi {
namespace nnet3 {

class NnetComputerFromEg {
 public:
  NnetComputerFromEg(const Nnet &nnet):
      nnet_(nnet), compiler_(nnet) { }

  // Compute the output (which will have the same number of rows as the number
  // of Indexes in the output of the eg), and put it in "output".
 // void Compute(const NnetExample &eg, Matrix<BaseFloat> *output) {
  void Compute(const NnetExample &eg, Vector<BaseFloat> *xvector) {
    ComputationRequest request;
    bool need_backprop = false, store_stats = false;
    GetComputationRequest(nnet_, eg, need_backprop, store_stats, &request);
    const NnetComputation &computation = *(compiler_.Compile(request));
    NnetComputeOptions options;
    if (GetVerboseLevel() >= 3)
      options.debug = true;
    NnetComputer computer(options, computation, nnet_, NULL);
    computer.AcceptInputs(nnet_, eg.io);
    computer.Run();
    const CuMatrixBase<BaseFloat> &nnet_output = computer.GetOutput("output");
    Matrix<BaseFloat> output(1, xvector->Dim());
    output.Resize(nnet_output.NumRows(), nnet_output.NumCols());
    nnet_output.CopyToMat(&output);
    xvector->CopyRowFromMat(output, 0);
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

    const char *usage =
      "Propagate the features through the network and write the output\n"
      "xvectors.  By default, xvectors are extracted once every\n"
      "--xvector-period using --chunk-size frames and output as an archive\n"
      "of matrices.  If --repeat=true, the xvectors are copied between\n"
      "periods, so that the output matrix has the same number of rows as\n"
      "the input.  If --output-as-vector=true, the xvectors are averaged\n"
      "across periods, and the output is a single vector for each utterance.\n"
      "\n"
      "Usage: nnet3-xvector-compute [options] <raw-nnet-in> "
      "<feats-rspecifier> <xvector-wspecifier>\n"
      " e.g.: nnet3-xvector-compute --xvector-period=50 final.raw "
      "scp:feats.scp ark:xvectors.ark\n";

    ParseOptions po(usage);
    Timer timer;

    NnetSimpleComputationOptions opts;
    std::string use_gpu = "yes";
    int32 chunk_size = 100;

    opts.Register(&po);

    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("chunk-size", &chunk_size,
      "Feature chunk size over which the xvector is computed.  "
      "If not set, defaults to xvector-period.");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    KALDI_ASSERT(chunk_size > 0);

    std::string nnet_rxfilename = po.GetArg(1),
                feat_rspecifier = po.GetArg(2),
                vector_wspecifier = po.GetArg(3);
    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);
    NnetComputerFromEg nnet_computer(nnet);

    BaseFloatVectorWriter vector_writer(vector_wspecifier);

    int32 num_success = 0,
          num_fail = 0,
          left_context,
          right_context,
          xvector_dim = nnet.OutputDim("output");
    int32 min_chunk_size = 100;
    int64 frame_count = 0;
    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string utt = feat_reader.Key();
      const Matrix<BaseFloat> &feats (feat_reader.Value());
      int32 num_rows = feats.NumRows(),
            feat_dim = feats.NumCols(),
            this_chunk_size = chunk_size;

      if (num_rows < min_chunk_size) {
        KALDI_WARN << "Minimum chunk size of " << min_chunk_size
                   << " is greater than the number of rows "
                   << "in utterance: " << utt;
        num_fail++;
        continue;
      } else if (num_rows < this_chunk_size) {
        KALDI_LOG << "Chunk size of " << this_chunk_size << " is greater than "
                  << "the number of rows in utterance: " << utt
                  << ", using chunk size  of " << num_rows;
        this_chunk_size = num_rows;
      }

      int32 num_chunks = ceil(num_rows / static_cast<BaseFloat>(chunk_size));

      Vector<BaseFloat> xvector_avg(xvector_dim, kSetZero);
      BaseFloat tot_weight = 0.0;

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
        Vector<BaseFloat> xvector(xvector_dim);
        tot_weight += offset;

        // Bundle the input features into an NnetExample.  Note that there are
        // more direct ways to do the computation, but this is easiest.
        NnetExample eg;
        NnetIo nnet_input = NnetIo("input", 0, sub_feats);
        for (std::vector<Index>::iterator indx_it = nnet_input.indexes.begin();
            indx_it != nnet_input.indexes.end(); ++indx_it)
          indx_it->n = 0;
        eg.io.push_back(nnet_input);
        // It doesn't matter what is in the output part of the NnetExample
        // (it isn't used) but the nnet computation will complain if it
        // isn't there.
        Posterior label(1);
        eg.io.push_back(NnetIo("output", xvector_dim, 0, label));
        if (sub_feats.NumRows() == chunk_size) {
          nnet_computer.Compute(eg, &xvector);
        // TODO: Segfaults seem to arise when the nnet_computer sees
        // too many unique computations.  So if the the number of feats
        // is not chunk_size (e.g., we're at the end of the features)
        // let's do the computation in a new nnet computer.
        // This is a temporary fix.  We should see why this is happening.
        } else {
          NnetComputerFromEg nnet_computer_alt(nnet);
          nnet_computer_alt.Compute(eg, &xvector);
        }
        xvector_avg.AddVec(offset, xvector);
      }

      xvector_avg.Scale(1.0 / tot_weight);
      vector_writer.Write(utt, xvector_avg);

      frame_count += feats.NumRows();
      num_success++;
    }

    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;

    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
