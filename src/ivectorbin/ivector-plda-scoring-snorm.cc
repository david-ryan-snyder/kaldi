// ivectorbin/ivector-plda-scoring-snorm.cc

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
        "NOTE: This binary is a work in progress!\n"
        "\n"
        "Similar to ivector-plda-scoring, but uses a pile of adaptation\n"
        "iVectors for score normalization (the first two arguments after\n"
        "the PLDA model).\n"
        "\n"
        "Computes log-likelihood ratios for trials using PLDA model\n"
        "Note: the 'trials-file' has lines of the form\n"
        "<key1> <key2>\n"
        "and the output will have the form\n"
        "<key1> <key2> [<dot-product>]\n"
        "(if either key could not be found, the dot-product field in the output\n"
        "will be absent, and this program will print a warning)\n"
        "For training examples, the input is the iVectors averaged over speakers;\n"
        "a separate archive containing the number of utterances per speaker may be\n"
        "optionally supplied using the --num-utts option; this affects the PLDA\n"
        "scoring (if not supplied, it defaults to 1 per speaker).\n"
        "\n"
        "Usage: ivector-plda-scoring-snorm <plda> <adapt-train-ivector-rspecifier>\n"
        " <adapt-test-ivector-rspecifier>  <train-ivector-rspecifier> <test-ivector-rspecifier>\n"
        " <trials-rxfilename> <scores-wxfilename>\n"
        "\n"
        "e.g.: ivector-plda-scoring --num-utts=ark:exp/train/num_utts.ark plda "
        "ark:exp/train/spk_ivectors.ark ark:exp/adaptation/ivectors.ark\n"
        " ark:exp/adaptation/ivectors.ark ark:exp/test/ivectors.ark trials scores\n"
        "See also: ivector-plda-scoring, ivector-compute-plda\n";

    ParseOptions po(usage);

    std::string num_utts_rspecifier;
    int32 max_comparisons = 1000;
    double top_percent = 10.0;

    PldaConfig plda_config;
    plda_config.Register(&po);
    po.Register("num-utts", &num_utts_rspecifier, "Table to read the number of "
                "utterances per speaker, e.g. ark:num_utts.ark\n");
    po.Register("top-percent", &top_percent,
      "Use the top percent of scores (usually 10%)");
    po.Register("max-comparisons", &max_comparisons,
      "Compare each train and test iVector with this many adaptation iVectors");

    po.Read(argc, argv);

    if (po.NumArgs() != 7) {
      po.PrintUsage();
      exit(1);
    }

    std::string plda_rxfilename = po.GetArg(1),
        train_ivector_snorm_rspecifier = po.GetArg(2),
        test_ivector_snorm_rspecifier = po.GetArg(3),
        train_ivector_rspecifier = po.GetArg(4),
        test_ivector_rspecifier = po.GetArg(5),
        trials_rxfilename = po.GetArg(6),
        scores_wxfilename = po.GetArg(7);

    //  diagnostics:
    double tot_test_renorm_scale = 0.0, tot_train_renorm_scale = 0.0,
      tot_test_renorm_scale_snorm = 0.0, tot_train_renorm_scale_snorm = 0.0;
    int64 num_train_ivectors = 0, num_train_errs = 0, num_test_ivectors = 0,
      num_train_ivectors_snorm = 0, num_test_ivectors_snorm = 0;

    int64 num_trials_done = 0, num_trials_err = 0;

    Plda plda;
    ReadKaldiObject(plda_rxfilename, &plda);

    int32 dim = plda.Dim();

    SequentialBaseFloatVectorReader train_ivector_reader(train_ivector_rspecifier);
    SequentialBaseFloatVectorReader test_ivector_reader(test_ivector_rspecifier);
    SequentialBaseFloatVectorReader train_ivector_snorm_reader(train_ivector_snorm_rspecifier);
    SequentialBaseFloatVectorReader test_ivector_snorm_reader(test_ivector_snorm_rspecifier);
    RandomAccessInt32Reader num_utts_reader(num_utts_rspecifier);

    typedef unordered_map<string, Vector<BaseFloat>*, StringHasher> HashType;
    typedef unordered_map<string, BaseFloat, StringHasher> FloatHash;

    // These hashes will contain the iVectors in the PLDA subspace
    // (that makes the within-class variance unit and diagonalizes the
    // between-class covariance).  They will also possibly be length-normalized,
    // depending on the config.
    HashType train_ivectors_snorm, test_ivectors_snorm, train_ivectors, test_ivectors;
    FloatHash means, stddevs;

    // TODO
    KALDI_LOG << "Reading test iVectors for snorm";
    for (; !test_ivector_snorm_reader.Done(); test_ivector_snorm_reader.Next()) {
      std::string spk = test_ivector_snorm_reader.Key();
      if (test_ivectors_snorm.count(spk) != 0) {
        KALDI_ERR << "Duplicate test snorm iVector found for speaker " << spk;
      }
      const Vector<BaseFloat> &ivector = test_ivector_snorm_reader.Value();
      Vector<BaseFloat> *transformed_ivector = new Vector<BaseFloat>(dim);

      tot_test_renorm_scale_snorm += plda.TransformIvector(plda_config, ivector,
                                                      1.0,
                                                      transformed_ivector);
      test_ivectors_snorm[spk] = transformed_ivector;
      num_test_ivectors_snorm++;
    }
    KALDI_LOG << "Read " << num_test_ivectors_snorm << " test snorm iVectors";
    if (num_test_ivectors_snorm == 0)
      KALDI_ERR << "No test snorm iVectors present.";
    KALDI_LOG << "Average renormalization scale on test snorm iVectors was "
              << (tot_test_renorm_scale_snorm / num_test_ivectors_snorm);
    // TODO
    KALDI_LOG << "Reading train iVectors for snorm";
    for (; !train_ivector_snorm_reader.Done(); train_ivector_snorm_reader.Next()) {
      std::string spk = train_ivector_snorm_reader.Key();
      if (train_ivectors_snorm.count(spk) != 0) {
        KALDI_ERR << "Duplicate training iVector found for speaker " << spk;
      }
      const Vector<BaseFloat> &ivector = train_ivector_snorm_reader.Value();
      Vector<BaseFloat> *transformed_ivector = new Vector<BaseFloat>(dim);

      tot_train_renorm_scale_snorm += plda.TransformIvector(plda_config, ivector,
                                                      1.0,
                                                      transformed_ivector);
      train_ivectors_snorm[spk] = transformed_ivector;
      num_train_ivectors_snorm++;
    }
    KALDI_LOG << "Read " << num_train_ivectors_snorm << " training snorm iVectors";
    if (num_train_ivectors_snorm == 0)
      KALDI_ERR << "No training sborm iVectors present.";
    KALDI_LOG << "Average renormalization scale on training snorm iVectors was "
              << (tot_train_renorm_scale_snorm / num_train_ivectors_snorm);
    // TODO
    KALDI_LOG << "Reading train iVectors";
    for (; !train_ivector_reader.Done(); train_ivector_reader.Next()) {
      std::string spk = train_ivector_reader.Key();
      if (train_ivectors.count(spk) != 0) {
        KALDI_ERR << "Duplicate training iVector found for speaker " << spk;
      }
      const Vector<BaseFloat> &ivector = train_ivector_reader.Value();
      int32 num_examples;
      if (!num_utts_rspecifier.empty()) {
        if (!num_utts_reader.HasKey(spk)) {
          KALDI_WARN << "Number of utterances not given for speaker " << spk;
          num_train_errs++;
          continue;
        }
        num_examples = num_utts_reader.Value(spk);
      } else {
        num_examples = 1;
      }
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


    Input ki(trials_rxfilename);
    bool binary = false;
    Output ko(scores_wxfilename, binary);

    double sum = 0.0, sumsq = 0.0;
    std::string line;

    // TODO
    for (HashType::iterator iter1 = train_ivectors.begin();
        iter1 != train_ivectors.end(); ++iter1) {
      string key1 = iter1->first;
      const Vector<BaseFloat> *train_ivector = train_ivectors[key1];
      Vector<double> train_ivector_dbl(*train_ivector);
      int32 num_train_examples;
      if (!num_utts_rspecifier.empty()) {
        // we already checked that it has this key.
        num_train_examples = num_utts_reader.Value(key1);
      } else {
        num_train_examples = 1;
      }
      std::vector<double> scores;
      std::vector<std::string> keys;
      for (HashType::iterator iter2 = train_ivectors_snorm.begin();
          iter2 != train_ivectors_snorm.end(); ++iter2) {
        string key2 = iter2->first;
        keys.push_back(key2);
      }
      std::random_shuffle(keys.begin(), keys.end());
      keys.resize(max_comparisons);
      for (int32 i = 0; i < keys.size(); i++) {
        const Vector<BaseFloat> *train_ivector_snorm = train_ivectors_snorm[keys[i]];
        Vector<double> train_ivector_snorm_dbl(*train_ivector_snorm);
        BaseFloat score = plda.LogLikelihoodRatio(train_ivector_dbl,
                                                num_train_examples,
                                                train_ivector_snorm_dbl);
        scores.push_back(score);
      }
      std::sort(scores.begin(), scores.end());
      std::reverse(scores.begin(), scores.end());
      int32 new_size = scores.size() * ((1.0 * top_percent) / 100.0);
      scores.resize(new_size); // Take the top 10% of the scores.
      double mean = 0.0;
      double stddev = 0.0;
      for (int32 i = 0; i < scores.size(); i++) {
        mean += scores[i];
        stddev += scores[i] * scores[i];
      }
      mean = (1.0 / scores.size()) * mean;
      stddev = sqrt((1.0 / scores.size()) * stddev - mean*mean);
      means[key1] = mean;
      stddevs[key1] = stddev;
    }
    // TODO
    for (HashType::iterator iter1 = test_ivectors.begin();
        iter1 != test_ivectors.end(); ++iter1) {
      string key1 = iter1->first;
      const Vector<BaseFloat> *test_ivector = test_ivectors[key1];
      Vector<double> test_ivector_dbl(*test_ivector);

      std::vector<double> scores;
      std::vector<std::string> keys;
      for (HashType::iterator iter2 = test_ivectors_snorm.begin();
          iter2 != test_ivectors_snorm.end(); ++iter2) {
        string key2 = iter2->first;
        keys.push_back(key2);
      }
      std::random_shuffle(keys.begin(), keys.end());
      keys.resize(max_comparisons);
      for (int32 i = 0; i < keys.size(); i++) {
        const Vector<BaseFloat> *test_ivector_snorm = test_ivectors_snorm[keys[i]];
        Vector<double> test_ivector_snorm_dbl(*test_ivector_snorm);
        BaseFloat score = plda.LogLikelihoodRatio(test_ivector_dbl, 1.0,
                                                test_ivector_snorm_dbl);
        scores.push_back(score);
      }
      std::sort(scores.begin(), scores.end());
      std::reverse(scores.begin(), scores.end());
      int32 new_size = scores.size() * ((1.0 * top_percent) / 100.0);
      scores.resize(new_size); // Take the top N% of the scores.
      double mean = 0.0;
      double stddev = 0.0;
      for (int32 i = 0; i < scores.size(); i++) {
        mean += scores[i];
        stddev += scores[i] * scores[i];
      }
      mean = (1.0 / scores.size()) * mean;
      stddev = sqrt((1.0 / scores.size()) * stddev - mean*mean);
      means[key1] = mean;
      stddevs[key1] = stddev;
    }

    // TODO
    while (std::getline(ki.Stream(), line)) {
      std::vector<std::string> fields;
      SplitStringToVector(line, " \t\n\r", true, &fields);
      if (fields.size() != 2) {
        KALDI_ERR << "Bad line " << (num_trials_done + num_trials_err)
                  << "in input (expected two fields: key1 key2): " << line;
      }
      std::string key1 = fields[0], key2 = fields[1];
      if (train_ivectors.count(key1) == 0) {
        KALDI_WARN << "Key " << key1 << " not present in training iVectors.";
        num_trials_err++;
        continue;
      }
      if (test_ivectors.count(key2) == 0) {
        KALDI_WARN << "Key " << key2 << " not present in test iVectors.";
        num_trials_err++;
        continue;
      }
      const Vector<BaseFloat> *train_ivector = train_ivectors[key1],
          *test_ivector = test_ivectors[key2];

      Vector<double> train_ivector_dbl(*train_ivector),
          test_ivector_dbl(*test_ivector);

      int32 num_train_examples;
      if (!num_utts_rspecifier.empty()) {
        // we already checked that it has this key.
        num_train_examples = num_utts_reader.Value(key1);
      } else {
        num_train_examples = 1;
      }

      // Raw score
      BaseFloat score = plda.LogLikelihoodRatio(train_ivector_dbl,
                                                num_train_examples,
                                                test_ivector_dbl);

      // After score normalization
      score = (score - means[key1]) / stddevs[key1] + (score - means[key2]) / stddevs[key2];

      sum += score;
      sumsq += score * score;
      num_trials_done++;
      ko.Stream() << key1 << ' ' << key2 << ' ' << score << std::endl;
    }

    for (HashType::iterator iter = train_ivectors_snorm.begin();
         iter != train_ivectors_snorm.end(); ++iter)
      delete iter->second;
    for (HashType::iterator iter = test_ivectors_snorm.begin();
         iter != test_ivectors_snorm.end(); ++iter)
      delete iter->second;

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
