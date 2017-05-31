// ivectorbin/agglomerative-cluster.cc

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
#include "util/stl-utils.h"
#include "tree/cluster-utils.h"
#include "tree/clusterable-classes.h"
#include "ivector/group-clusterable.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
      "TODO: This binary is a work in progress\n"
      "\n"
      "Cluster a list of scores.  The first argument is a file where each\n"
      "line looks like this:\n"
      "    <utt-id1> <utt-id2> score\n"
      "Usage: agglomerative-cluster [options] <scores-rspecifier> "
      "<spk2utt-rspecifier> <labels-wspecifier>\n"
      "e.g.: \n"
      " agglomerative-cluster ark:scores.ark ark:spk2utt \n"
      "   ark,t:labels.txt\n";

    ParseOptions po(usage);
    BaseFloat threshold = 0.0;
    int32 num_speakers = -1;

    po.Register("num-speakers", &num_speakers, "Optionally specify the desired"
      "number of clusters.  If left unspecified, it uses the threshold.");
    po.Register("threshold", &threshold, "Merging clusters if their distance"
      "is less than this threshold.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string scores_rxfilename = po.GetArg(1),
      utt2spk_wspecifier = po.GetArg(2);
    Input ki(scores_rxfilename);
    Int32Writer utt2spk_writer(utt2spk_wspecifier);

    bool binary = false;

    double sum = 0.0, sumsq = 0.0;
    std::string line;

    std::vector<std::string> utts;
    unordered_map<std::string, int32, StringHasher> string2int;
    unordered_map<std::pair<int32, int32>, BaseFloat,
        PairHasher<int32, int32> > scores;

    BaseFloat min_distance = std::numeric_limits<BaseFloat>::max(),
        max_distance = std::numeric_limits<BaseFloat>::min();
    while (std::getline(ki.Stream(), line)) {
      std::vector<std::string> fields;
      SplitStringToVector(line, " \t\n\r", true, &fields);
      if (fields.size() != 3) {
        KALDI_ERR << "Bad line " << line;
      }
      BaseFloat score;
      std::string key1 = fields[0], key2 = fields[1];

      if (!ConvertStringToReal(fields[2], &score)) {
        KALDI_ERR << "Invalid input line (third field must be float): "
                  << line;
      }

      int32 int_key1, int_key2;
      if (string2int.find(key1) == string2int.end()) {
        string2int[key1] = utts.size();
        utts.push_back(key1);
      }

      if (string2int.find(key2) == string2int.end()) {
        string2int[key2] = utts.size();
        utts.push_back(key2);
      }

      int_key1 = string2int[key1];
      int_key2 = string2int[key2];

      score = -score;
      scores[std::pair<int32, int32>(int_key1, int_key2)] = score;
      scores[std::pair<int32, int32>(int_key2, int_key1)] = score;

      if (score > max_distance) {
        max_distance = score;
      }
      if (score < min_distance) {
        min_distance = score;
      }
    }

    // Shift negative scores to >= 0
    if (min_distance < 0.0) {
      max_distance = max_distance - min_distance;
      for (unordered_map<std::pair<int32, int32>, BaseFloat,
          PairHasher<int32, int32> >::iterator itr = scores.begin();
          itr != scores.end(); itr++) {
        itr->second = itr->second - min_distance;
      }
      threshold = -threshold - min_distance;
    }

    std::vector<Clusterable*> clusterables;
    std::vector<int32> spk_ids;

    for (int32 i = 0; i < utts.size(); i++) {
      std::set<int32> points;
      points.insert(i);
      // The argument max_distance is there so that we know what value
      // to give to missing entries in scores.  TODO: this should probably
      // be provided as an argument in some GroupClusterableConfig.
      clusterables.push_back(new GroupClusterable(points, &scores,
        max_distance));
    }

    // Either use a known number of speakers when clustering,
    // or use the stopping threshold.  ClusterBottomUp is equivalent
    // to agglomerative clustering.
    if (num_speakers != -1) {
      ClusterBottomUp(clusterables, std::numeric_limits<BaseFloat>::max(),
        num_speakers, NULL, &spk_ids);
    } else {
      ClusterBottomUp(clusterables, threshold, 1, NULL, &spk_ids);
    }

    KALDI_ASSERT(spk_ids.size() == utts.size());
    // Write out the speakers that the utterances correspond to.
    for (int32 i = 0; i < spk_ids.size(); i++) {
        KALDI_LOG << utts[i] << " " << spk_ids[i];
        utt2spk_writer.Write(utts[i], spk_ids[i]);
    }

    DeletePointers(&clusterables);
    return 0;

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
