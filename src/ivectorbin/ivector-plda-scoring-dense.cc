// ivectorbin/ivector-plda-scoring-dense.cc

// Copyright 2013  Daniel Povey
//           2017  David Snyder

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
  typedef std::string string;
  try {
    const char *usage =
        "TODO: this is a work in progress.  It is possible that this\n"
        "functionality might not even need to be included in a stand alone\n"
        "binary.\n"
        "\n"
        "Computes log-likelihood ratios for all pairs of iVectors in the\n"
        "first and second rspecifiers (which might be the same file!)\n"
        "using a PLDA model.  This differs from ivector-plda-scoring in that\n"
        "we do not use a trials file.  The output will have the form\n"
        "<key1> <key2> [<plda-score>]\n"
        "\n"
        "Usage: ivector-plda-scoring-dense <plda> <ivector-rspecifier1>\n"
        "  <ivector-rspecifier2> <scores-wxfilename>\n"
        "\n"
        "e.g.: ivector-plda-scoring-dense plda "
        "ark:exp/train/ivectors.ark ark:exp/test/ivectors.ark scores\n"
        "See also: ivector-plda-scoring, ivector-compute-dot-products,\n"
        "  ivector-compute-plda\n";

    ParseOptions po(usage);

    BaseFloat min_score = 0.0;
    PldaConfig plda_config;
    plda_config.Register(&po);
    po.Register("min-score", &min_score, "If the score is below this value,"
        "don't print it.");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string plda_rxfilename = po.GetArg(1),
        train_ivector_rspecifier = po.GetArg(2),
        test_ivector_rspecifier = po.GetArg(3),
        scores_wxfilename = po.GetArg(4);

    //  diagnostics:
    double tot_test_renorm_scale = 0.0, tot_train_renorm_scale = 0.0;
    int64 num_train_ivectors = 0, num_train_errs = 0, num_test_ivectors = 0;

    int64 num_trials_done = 0, num_trials_err = 0;

    Plda plda;
    ReadKaldiObject(plda_rxfilename, &plda);

    int32 dim = plda.Dim();

    SequentialBaseFloatVectorReader train_ivector_reader(train_ivector_rspecifier);
    SequentialBaseFloatVectorReader test_ivector_reader(test_ivector_rspecifier);

    typedef unordered_map<string, Vector<BaseFloat>*, StringHasher> HashType;

    // These hashes will contain the iVectors in the PLDA subspace
    // (that makes the within-class variance unit and diagonalizes the
    // between-class covariance).  They will also possibly be length-normalized,
    // depending on the config.
    HashType train_ivectors, test_ivectors;

    KALDI_LOG << "Reading train iVectors";
    for (; !train_ivector_reader.Done(); train_ivector_reader.Next()) {
      std::string spk = train_ivector_reader.Key();
      if (train_ivectors.count(spk) != 0) {
        KALDI_ERR << "Duplicate training iVector found for speaker " << spk;
      }
      const Vector<BaseFloat> &ivector = train_ivector_reader.Value();
      int32 num_examples = 1; // this value is always used for test (affects the
                              // length normalization in the TransformIvector
                              // function).
      Vector<BaseFloat> *transformed_ivector = new Vector<BaseFloat>(dim);

      tot_train_renorm_scale += plda.TransformIvector(plda_config, ivector,
                                                      num_examples,
                                                      transformed_ivector);
      train_ivectors[spk] = transformed_ivector;
      num_train_ivectors++;
    }
    KALDI_LOG << "Read " << num_train_ivectors << " training iVectors, "
              << "errors on " << num_train_errs;
    if (num_train_ivectors == 0)
      KALDI_ERR << "No training iVectors present.";
    KALDI_LOG << "Average renormalization scale on training iVectors was "
              << (tot_train_renorm_scale / num_train_ivectors);

    KALDI_LOG << "Reading test iVectors";
    for (; !test_ivector_reader.Done(); test_ivector_reader.Next()) {
      std::string utt = test_ivector_reader.Key();
      if (test_ivectors.count(utt) != 0) {
        KALDI_ERR << "Duplicate test iVector found for utterance " << utt;
      }
      const Vector<BaseFloat> &ivector = test_ivector_reader.Value();
      int32 num_examples = 1; // this value is always used for test (affects the
                              // length normalization in the TransformIvector
                              // function).
      Vector<BaseFloat> *transformed_ivector = new Vector<BaseFloat>(dim);

      tot_test_renorm_scale += plda.TransformIvector(plda_config, ivector,
                                                     num_examples,
                                                     transformed_ivector);
      test_ivectors[utt] = transformed_ivector;
      num_test_ivectors++;
    }
    KALDI_LOG << "Read " << num_test_ivectors << " test iVectors.";
    if (num_test_ivectors == 0)
      KALDI_ERR << "No test iVectors present.";
    KALDI_LOG << "Average renormalization scale on test iVectors was "
              << (tot_test_renorm_scale / num_test_ivectors);


    bool binary = false;
    Output ko(scores_wxfilename, binary);

    double sum = 0.0, sumsq = 0.0;
    for (HashType::iterator iter1 = train_ivectors.begin();
         iter1 != train_ivectors.end(); ++iter1) {
      const string key1 = iter1->first;
      const Vector<BaseFloat> *train_ivector = train_ivectors[key1];
      Vector<double> train_ivector_dbl(*train_ivector);
      for (HashType::iterator iter2 = test_ivectors.begin();
           iter2 != test_ivectors.end(); ++iter2) {
        const string key2 = iter2->first;
        if (!key1.compare(key2))
          continue;
        const Vector<BaseFloat> *test_ivector = test_ivectors[key2];
        Vector<double> test_ivector_dbl(*test_ivector);

        int32 num_train_examples = 1;
        BaseFloat score = plda.LogLikelihoodRatio(train_ivector_dbl,
                                                  num_train_examples,
                                                  test_ivector_dbl);
        if (score > min_score) {
          sum += score;
          sumsq += score * score;
          num_trials_done++;
          ko.Stream() << key1 << ' ' << key2 << ' ' << score << std::endl;
        }
      }
    }

    for (HashType::iterator iter = train_ivectors.begin();
         iter != train_ivectors.end(); ++iter)
      delete iter->second;
    for (HashType::iterator iter = test_ivectors.begin();
         iter != test_ivectors.end(); ++iter)
      delete iter->second;


    if (num_trials_done != 0) {
      BaseFloat mean = sum / num_trials_done, scatter = sumsq / num_trials_done,
          variance = scatter - mean * mean, stddev = sqrt(variance);
      KALDI_LOG << "Mean score was " << mean << ", standard deviation was "
                << stddev;
    }
    KALDI_LOG << "Processed " << num_trials_done << " trials, " << num_trials_err
              << " had errors.";
    return (num_trials_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
