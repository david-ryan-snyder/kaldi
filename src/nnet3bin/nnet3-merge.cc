// nnet3bin/nnet3-merge.cc

// Copyright 2018  David Snyder

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
#include "hmm/transition-model.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;

    const char *usage =
        "Merge two nnet3 neural networks, so that they share the same input\n"
        "node.  Use the option --suffix to add a suffix to the second\n"
        "network, prior to merging, so that there are no conflicts with\n"
        "the first network.  Use nnet3-copy to perform further\n"
        "modifications, such as appending or summing the output of the two\n"
        "networks\n"
        "\n"
        "Usage:  nnet3-merge [options] <nnet-in1> <nnet-in2> <nnet-out>\n"
        "e.g.:\n"
        " nnet3-merge --suffix=-2 final1.raw final2.raw out.raw\n";

    bool binary_write = true;
    std::string suffix = "-2";

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("suffix", &suffix,
                "Suffix to add to the node and component node names of nnet-in2.");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string raw_nnet_rxfilename1 = po.GetArg(1),
                raw_nnet_rxfilename2 = po.GetArg(2),
                raw_nnet_wxfilename = po.GetArg(3);

    Nnet nnet1, nnet2;
    ReadKaldiObject(raw_nnet_rxfilename1, &nnet1);
    ReadKaldiObject(raw_nnet_rxfilename2, &nnet2);
    AddSuffixToNodes(suffix, &nnet2);

    for (int32 i = 0; i < nnet2.NumComponents(); i++) {
      Component *c = nnet2.GetComponent(i);
      std::string component_name = nnet2.GetComponentName(i);
      nnet1.AddComponent(component_name, c->Copy());
    }

    std::ostringstream config;
    for (int32 i = 0; i < nnet2.NumNodes(); i++) {
      if (!nnet2.IsComponentInputNode(i)) {
        std::string node_name = nnet2.GetNodeName(i);
        if (node_name != "input" && node_name != "output") {
          config << nnet2.GetAsConfigLine(i, false) << "\n";
        }
      }
    }

    std::istringstream is(config.str());
    nnet1.ReadConfig(is);

    WriteKaldiObject(nnet1, raw_nnet_wxfilename, binary_write);
    KALDI_LOG << "Merged raw neural nets " << raw_nnet_rxfilename1
              << " and " << raw_nnet_rxfilename2 << " to "
              << raw_nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
