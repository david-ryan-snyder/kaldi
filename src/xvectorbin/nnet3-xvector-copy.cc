// xvectorbin/nnet3-xvector-copy.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)
//           2015  Xingyu Na
//           2016  David Snyder

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

#include <typeinfo>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-utils.h"
#include "nnet3/nnet-simple-component.h"

namespace kaldi {
void RenameNodes(const std::string nodes_to_rename, nnet3::Nnet *nnet) {
  std::vector<std::string> name_pairs;
  SplitStringToVector(nodes_to_rename, ",", false, &name_pairs);
  for (int32 i = 0; i < name_pairs.size(); i++) {
    std::vector<std::string> name_pair;
    SplitStringToVector(name_pairs[i], ":", false, &name_pair);
    if (name_pair.size() != 2)
      KALDI_ERR << "Malformed argument to option --rename-output-nodes";
    nnet->RenameOutputNode(name_pair[0], name_pair[1]);
  }
}

void FloorS(BaseFloat floor, nnet3::Nnet *nnet) {
  nnet3::ConstantFunctionComponent *s_comp
    = dynamic_cast<nnet3::ConstantFunctionComponent *>(
    nnet->GetComponent(nnet->GetComponentIndex("x-s")));
  int32 s_dim = s_comp->OutputDim();
  Vector<BaseFloat> s_vec(s_dim);
  s_comp->Vectorize(&s_vec);
  KALDI_LOG << "s_dim = " << s_dim;
  int32 d = (0.5) * (1 + sqrt(1 + 8 * s_dim)) - 1;
  if (((d + 1) * d) / 2 != s_dim)
    KALDI_ERR << "Output 'x-s' cannot be interpreted as a symmetric matrix.";
  SpMatrix<BaseFloat> S(d);
  //GetConstantOutput(*nnet, "x-s", &s_vec);
  S.CopyFromVec(s_vec);
  S.ApplyFloor(floor);
  SubVector<BaseFloat> s_vec_floored(S.Data(), s_dim);
  //nnet3::ConstantFunctionComponent *s_comp
  //  = dynamic_cast<nnet3::ConstantFunctionComponent *>(
  //  nnet->GetComponent(nnet->GetComponentIndex("x-s")));
  s_comp->UnVectorize(s_vec_floored);
}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;

    const char *usage =
        "TODO Copy xvector nnet3 neural network to standard output\n"
        "Also supports setting all the learning rates to a value\n"
        "(the --learning-rate option)\n"
        "\n"
        "Usage:  nnet3-xvector-copy [options] <nnet-in> <nnet-out>\n"
        "e.g.:\n"
        " nnet3-copy --binary=false 0.raw text.raw\n";

    bool binary_write = true;
    std::string learning_rates_str = "";
    std::string rename_output_nodes = "";
    BaseFloat learning_rate = -1,
      dropout = 0.0,
      floor_s = 0.0;
    std::string nnet_config, edits_config, edits_str;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("learning-rate", &learning_rate,
                "If supplied, all the learning rates of updatable components"
                "are set to this value.");
    po.Register("rename-output-nodes", &rename_output_nodes, "Rename one or"
                " more output nodes. For example, TODO");
    po.Register("scale-learning-rates", &learning_rates_str, "TODO");
    po.Register("nnet-config", &nnet_config,
                "Name of nnet3 config file that can be used to add or replace "
                "components or nodes of the neural network (the same as you "
                "would give to nnet3-init).");
    po.Register("edits-config", &edits_config,
                "Name of edits-config file that can be used to modify the network "
                "(applied after nnet-config).  See comments for ReadEditConfig()"
                "in nnet3/nnet-utils.h to see currently supported commands.");
    po.Register("edits", &edits_str,
                "Can be used as an inline alternative to edits-config; semicolons "
                "will be converted to newlines before parsing.  E.g. "
                "'--edits=remove-orphans'.");
    po.Register("set-dropout-proportion", &dropout, "Set dropout proportion "
                "in all DropoutComponent to this value.");
    po.Register("floor-s", &floor_s, "Floor eigenvalues "
                "of symmetric matrix S to this value.");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string raw_nnet_rxfilename = po.GetArg(1),
                raw_nnet_wxfilename = po.GetArg(2);

    Nnet nnet;
    ReadKaldiObject(raw_nnet_rxfilename, &nnet);
    if (rename_output_nodes != "")
      RenameNodes(rename_output_nodes, &nnet);

    if (!nnet_config.empty()) {
      Input ki(nnet_config);
      nnet.ReadConfig(ki.Stream());
    }

    FloorS(floor_s, &nnet);

    if (learning_rate >= 0)
      SetLearningRate(learning_rate, &nnet);

    if (dropout > 0)
      SetDropoutProportion(dropout, &nnet);

    if (!edits_config.empty()) {
      Input ki(edits_config);
      ReadEditConfig(ki.Stream(), &nnet);
    }
    if (!edits_str.empty()) {
      for (size_t i = 0; i < edits_str.size(); i++)
        if (edits_str[i] == ';')
          edits_str[i] = '\n';
      std::istringstream is(edits_str);
      ReadEditConfig(is, &nnet);
    }

    if (learning_rates_str != "") {
      std::vector<BaseFloat> learning_rates_std_vec;
      SplitStringToFloats(learning_rates_str, ",", false, &learning_rates_std_vec);
      Vector<BaseFloat> learning_rates(learning_rates_std_vec.size());
      for (int32 i = 0; i < learning_rates_std_vec.size(); i++)
        learning_rates(i) = learning_rates_std_vec[i];
      ScaleLearningRates(learning_rates, &nnet);
    }

    WriteKaldiObject(nnet, raw_nnet_wxfilename, binary_write);
    KALDI_LOG << "Copied raw neural net from " << raw_nnet_rxfilename
              << " to " << raw_nnet_wxfilename;

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
