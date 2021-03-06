// nnet3/nnet-test-utils.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)
// Copyright      2015  Johns Hopkins University (author: Vijayaditya Peddinti)

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

#include <iterator>
#include <sstream>
#include "nnet3/nnet-test-utils.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {


// A super-simple case that is just a single affine component, no nonlinearity,
// no splicing.
void GenerateConfigSequenceSimplest(
    const NnetGenerationOptions &opts,
    std::vector<std::string> *configs) {
  std::ostringstream os;

  int32 input_dim = 10 + Rand() % 20,
       output_dim = 100 + Rand() % 200;


  os << "component name=affine1 type=AffineComponent input-dim="
     << input_dim << " output-dim=" << output_dim << std::endl;

  os << "input-node name=input dim=" << input_dim << std::endl;
  os << "component-node name=affine1_node component=affine1 input=input\n";
  os << "output-node name=output input=affine1_node\n";
  configs->push_back(os.str());
}

// A setup with context and an affine component, but no nonlinearity.
void GenerateConfigSequenceSimpleContext(
    const NnetGenerationOptions &opts,
    std::vector<std::string> *configs) {
  std::ostringstream os;

  std::vector<int32> splice_context;
  for (int32 i = -5; i < 4; i++)
    if (Rand() % 3 == 0)
      splice_context.push_back(i);
  if (splice_context.empty())
    splice_context.push_back(0);

  int32 input_dim = 10 + Rand() % 20,
      spliced_dim = input_dim * splice_context.size(),
       output_dim = 100 + Rand() % 200;

  os << "component name=affine1 type=AffineComponent input-dim="
     << spliced_dim << " output-dim=" << output_dim << std::endl;

  os << "input-node name=input dim=" << input_dim << std::endl;

  os << "component-node name=affine1_node component=affine1 input=Append(";
  for (size_t i = 0; i < splice_context.size(); i++) {
    int32 offset = splice_context[i];
    os << "Offset(input, " << offset << ")";
    if (i + 1 < splice_context.size())
      os << ", ";
  }
  os << ")\n";
  os << "output-node name=output input=affine1_node\n";
  configs->push_back(os.str());
}



// A simple case, just to get started.
// Generate a single config with one input, splicing, and one hidden layer.
// Also sometimes generate a part of the config that adds a new hidden layer.
void GenerateConfigSequenceSimple(
    const NnetGenerationOptions &opts,
    std::vector<std::string> *configs) {
  std::ostringstream os;

  std::vector<int32> splice_context;
  for (int32 i = -5; i < 4; i++)
    if (Rand() % 3 == 0)
      splice_context.push_back(i);
  if (splice_context.empty())
    splice_context.push_back(0);

  int32 input_dim = 10 + Rand() % 20,
      spliced_dim = input_dim * splice_context.size(),
      output_dim = 100 + Rand() % 200,
      hidden_dim = 40 + Rand() % 50;
  bool use_final_nonlinearity = (opts.allow_final_nonlinearity &&
                                 RandInt(0, 1) == 0);
  os << "component name=affine1 type=NaturalGradientAffineComponent input-dim="
     << spliced_dim << " output-dim=" << hidden_dim << std::endl;
  os << "component name=relu1 type=RectifiedLinearComponent dim="
     << hidden_dim << std::endl;
  os << "component name=final_affine type=NaturalGradientAffineComponent input-dim="
     << hidden_dim << " output-dim=" << output_dim << std::endl;
  if (use_final_nonlinearity) {
    if (Rand() % 2 == 0) {
      os << "component name=logsoftmax type=SoftmaxComponent dim="
         << output_dim << std::endl;
    } else {
      os << "component name=logsoftmax type=LogSoftmaxComponent dim="
         << output_dim << std::endl;
    }
  }
  os << "input-node name=input dim=" << input_dim << std::endl;

  os << "component-node name=affine1_node component=affine1 input=Append(";
  for (size_t i = 0; i < splice_context.size(); i++) {
    int32 offset = splice_context[i];
    os << "Offset(input, " << offset << ")";
    if (i + 1 < splice_context.size())
      os << ", ";
  }
  os << ")\n";
  os << "component-node name=nonlin1 component=relu1 input=affine1_node\n";
  os << "component-node name=final_affine component=final_affine input=nonlin1\n";
  if (use_final_nonlinearity) {
    os << "component-node name=output_nonlin component=logsoftmax input=final_affine\n";
    os << "output-node name=output input=output_nonlin\n";
  } else {
    os << "output-node name=output input=final_affine\n";
  }
  configs->push_back(os.str());

  if ((Rand() % 2) == 0) {
    std::ostringstream os2;
    os2 << "component name=affine2 type=NaturalGradientAffineComponent input-dim="
        << hidden_dim << " output-dim=" << hidden_dim << std::endl;
    os2 << "component name=relu2 type=RectifiedLinearComponent dim="
        << hidden_dim << std::endl;
    // regenerate the final_affine component when we add the new config.
    os2 << "component name=final_affine type=NaturalGradientAffineComponent input-dim="
        << hidden_dim << " output-dim=" << output_dim << std::endl;
    os2 << "component-node name=affine2 component=affine2 input=nonlin1\n";
    os2 << "component-node name=relu2 component=relu2 input=affine2\n";
    os2 << "component-node name=final_affine component=final_affine input=relu2\n";
    configs->push_back(os2.str());
  }
}


// This generates a single config corresponding to an RNN.
void GenerateConfigSequenceRnn(
    const NnetGenerationOptions &opts,
    std::vector<std::string> *configs) {
  std::ostringstream os;

  std::vector<int32> splice_context;
  for (int32 i = -5; i < 4; i++)
    if (Rand() % 3 == 0)
      splice_context.push_back(i);
  if (splice_context.empty())
    splice_context.push_back(0);

  int32 input_dim = 10 + Rand() % 20,
      spliced_dim = input_dim * splice_context.size(),
      output_dim = 100 + Rand() % 200,
      hidden_dim = 40 + Rand() % 50;
  os << "component name=affine1 type=NaturalGradientAffineComponent input-dim="
     << spliced_dim << " output-dim=" << hidden_dim << std::endl;
  if (RandInt(0, 1) == 0) {
    os << "component name=nonlin1 type=RectifiedLinearComponent dim="
       << hidden_dim << std::endl;
  } else {
    os << "component name=nonlin1 type=TanhComponent dim="
       << hidden_dim << std::endl;
  }
  os << "component name=recurrent_affine1 type=NaturalGradientAffineComponent input-dim="
     << hidden_dim << " output-dim=" << hidden_dim << std::endl;
  os << "component name=affine2 type=NaturalGradientAffineComponent input-dim="
     << hidden_dim << " output-dim=" << output_dim << std::endl;
  os << "component name=logsoftmax type=LogSoftmaxComponent dim="
     << output_dim << std::endl;
  os << "input-node name=input dim=" << input_dim << std::endl;

  os << "component-node name=affine1_node component=affine1 input=Append(";
  for (size_t i = 0; i < splice_context.size(); i++) {
    int32 offset = splice_context[i];
    os << "Offset(input, " << offset << ")";
    if (i + 1 < splice_context.size())
      os << ", ";
  }
  os << ")\n";
  os << "component-node name=recurrent_affine1 component=recurrent_affine1 "
        "input=Offset(nonlin1, -1)\n";
  os << "component-node name=nonlin1 component=nonlin1 "
        "input=Sum(affine1_node, IfDefined(recurrent_affine1))\n";
  os << "component-node name=affine2 component=affine2 input=nonlin1\n";
  os << "component-node name=output_nonlin component=logsoftmax input=affine2\n";
  os << "output-node name=output input=output_nonlin\n";
  configs->push_back(os.str());
}



// This generates a config sequence for what I *think* is a clockwork RNN, in
// that different parts operate at different speeds.  The output layer is
// evaluated every frame, but the internal RNN layer is evaluated every 3
// frames.
void GenerateConfigSequenceRnnClockwork(
    const NnetGenerationOptions &opts,
    std::vector<std::string> *configs) {
  std::ostringstream os;

  std::vector<int32> splice_context;
  for (int32 i = -5; i < 4; i++)
    if (Rand() % 3 == 0)
      splice_context.push_back(i);
  if (splice_context.empty())
    splice_context.push_back(0);

  int32 input_dim = 10 + Rand() % 20,
      spliced_dim = input_dim * splice_context.size(),
      output_dim = 100 + Rand() % 200,
      hidden_dim = 40 + Rand() % 50;
  os << "component name=affine1 type=NaturalGradientAffineComponent input-dim="
     << spliced_dim << " output-dim=" << hidden_dim << std::endl;
  os << "component name=nonlin1 type=RectifiedLinearComponent dim="
     << hidden_dim << std::endl;
  os << "component name=recurrent_affine1 type=NaturalGradientAffineComponent input-dim="
     << hidden_dim << " output-dim=" << hidden_dim << std::endl;
  // the suffix _0, _1, _2 equals the index of the output-frame modulo 3; there
  // are 3 versions of the final affine layer.  There was a paper by Vincent
  // Vanhoucke about something like this.
  os << "component name=final_affine_0 type=NaturalGradientAffineComponent input-dim="
     << hidden_dim << " output-dim=" << output_dim << std::endl;
  os << "component name=final_affine_1 type=NaturalGradientAffineComponent input-dim="
     << hidden_dim << " output-dim=" << output_dim << std::endl;
  os << "component name=final_affine_2 type=NaturalGradientAffineComponent input-dim="
     << hidden_dim << " output-dim=" << output_dim << std::endl;
  os << "component name=logsoftmax type=LogSoftmaxComponent dim="
     << output_dim << std::endl;
  os << "input-node name=input dim=" << input_dim << std::endl;

  os << "component-node name=affine1_node component=affine1 input=Append(";
  for (size_t i = 0; i < splice_context.size(); i++) {
    int32 offset = splice_context[i];
    os << "Offset(input, " << offset << ")";
    if (i + 1 < splice_context.size())
      os << ", ";
  }
  os << ")\n";
  os << "component-node name=recurrent_affine1 component=recurrent_affine1 "
        "input=Offset(nonlin1, -1)\n";
  os << "component-node name=nonlin1 component=nonlin1 "
        "input=Sum(affine1_node, IfDefined(recurrent_affine1))\n";
  os << "component-node name=final_affine_0 component=final_affine_0 input=nonlin1\n";
  os << "component-node name=final_affine_1 component=final_affine_1 input=Offset(nonlin1, -1)\n";
  os << "component-node name=final_affine_2 component=final_affine_2 input=Offset(nonlin1, 1)\n";
  os << "component-node name=output_nonlin component=logsoftmax input=Switch(final_affine_0, final_affine_1, final_affine_2)\n";
  os << "output-node name=output input=output_nonlin\n";
  configs->push_back(os.str());
}



// This generates a single config corresponding to an LSTM.
// based on the equations in
// Sak et. al. "LSTM based RNN architectures for LVCSR", 2014
// We name the components based on the following equations (Eqs 7-15 in paper)
//      i(t) = S(Wix * x(t) + Wir * r(t-1) + Wic * c(t-1) + bi)
//      f(t) = S(Wfx * x(t) + Wfr * r(t-1) + Wfc * c(t-1) + bf)
//      c(t) = {f(t) .* c(t-1)} + {i(t) .* g(Wcx * x(t) + Wcr * r(t-1) + bc)}
//      o(t) = S(Wox * x(t) + Wor * r(t-1) + Woc * c(t) + bo)
//      m(t) = o(t) .* h(c(t))
//      r(t) = Wrm * m(t)
//      p(t) = Wpm * m(t)
//      y(t) = Wyr * r(t) + Wyp * p(t) + by
// where S : sigmoid
// matrix with feed-forward connections
// from the input x(t)
// W*x = [Wix^T, Wfx^T, Wcx^T, Wox^T]^T

// matrix with recurrent (feed-back) connections
// from the output projection
// W*r = [Wir^T, Wfr^T, Wcr^T, Wor^T]^T

// matrix to generate r(t) and p(t)
// m(t)
// W*m = [Wrm^T, Wpm^T]^T
// matrix to generate y(t)
// Wy* = [Wyr^T, Wyp^T]

// Diagonal matrices with recurrent connections and feed-forward connections
// from the cell output c(t) since these can be both recurrent and
// feed-forward we dont combine the matrices
// Wic, Wfc, Woc


void GenerateConfigSequenceLstm(
    const NnetGenerationOptions &opts,
    std::vector<std::string> *configs) {
  std::ostringstream os;

  std::vector<int32> splice_context;
  for (int32 i = -5; i < 4; i++)
    if (Rand() % 3 == 0)
      splice_context.push_back(i);
  if (splice_context.empty())
    splice_context.push_back(0);

  int32 input_dim = 10 + Rand() % 20,
      spliced_dim = input_dim * splice_context.size(),
      output_dim = 100 + Rand() % 200,
      cell_dim = 40 + Rand() % 50,
      projection_dim = std::ceil(cell_dim / (Rand() % 10 + 1));

  os << "input-node name=input dim=" << input_dim << std::endl;

  // Parameter Definitions W*(* replaced by - to have valid names)
  // Input gate control : Wi* matrices
  os << "component name=Wi-xr type=NaturalGradientAffineComponent"
     << " input-dim=" << spliced_dim + projection_dim
     << " output-dim=" << cell_dim << std::endl;
  os << "component name=Wic type=PerElementScaleComponent "
     << " dim=" << cell_dim << std::endl;

  // Forget gate control : Wf* matrices
  os << "component name=Wf-xr type=NaturalGradientAffineComponent"
     << " input-dim=" << spliced_dim + projection_dim
     << " output-dim=" << cell_dim << std::endl;
  os << "component name=Wfc type=PerElementScaleComponent "
     << " dim=" << cell_dim << std::endl;

  // Output gate control : Wo* matrices
  os << "component name=Wo-xr type=NaturalGradientAffineComponent"
     << " input-dim=" << spliced_dim + projection_dim
     << " output-dim=" << cell_dim  << std::endl;
  os << "component name=Woc type=PerElementScaleComponent "
     << " dim=" << cell_dim << std::endl;

  // Cell input matrices : Wc* matrices
  os << "component name=Wc-xr type=NaturalGradientAffineComponent"
     << " input-dim=" << spliced_dim + projection_dim
     << " output-dim=" << cell_dim  << std::endl;



  // projection matrices : Wrm and Wpm
  os << "component name=W-m type=NaturalGradientAffineComponent "
     << " input-dim=" << cell_dim
     << " output-dim=" << 2 * projection_dim << std::endl;

  // Output : Wyr and Wyp
  os << "component name=Wy- type=NaturalGradientAffineComponent "
     << " input-dim=" << 2 * projection_dim
     << " output-dim=" << cell_dim << std::endl;

  // Defining the diagonal matrices
  // Defining the final affine transform
  os << "component name=final_affine type=NaturalGradientAffineComponent "
     << "input-dim=" << cell_dim << " output-dim=" << output_dim << std::endl;
  os << "component name=logsoftmax type=LogSoftmaxComponent dim="
     << output_dim << std::endl;

  // Defining the non-linearities
  //  declare a no-op component so that we can use a sum descriptor's output
  //  multiple times, and to make the config more readable given the equations
  os << "component name=i type=SigmoidComponent dim="
     << cell_dim << std::endl;
  os << "component name=f type=SigmoidComponent dim="
     << cell_dim << std::endl;
  os << "component name=o type=SigmoidComponent dim="
     << cell_dim << std::endl;
  os << "component name=g type=TanhComponent dim="
     << cell_dim << std::endl;
  os << "component name=h type=TanhComponent dim="
     << cell_dim << std::endl;
  os << "component name=c1 type=ElementwiseProductComponent "
     << " input-dim=" << 2 * cell_dim
     << " output-dim=" << cell_dim << std::endl;
  os << "component name=c2 type=ElementwiseProductComponent "
     << " input-dim=" << 2 * cell_dim
     << " output-dim=" << cell_dim << std::endl;
  os << "component name=m type=ElementwiseProductComponent "
     << " input-dim=" << 2 * cell_dim
     << " output-dim=" << cell_dim << std::endl;

  // Defining the computations
  std::ostringstream temp_string_stream;
  for (size_t i = 0; i < splice_context.size(); i++) {
    int32 offset = splice_context[i];
    temp_string_stream << "Offset(input, " << offset << ")";
    if (i + 1 < splice_context.size())
      temp_string_stream << ", ";
  }
  std::string spliced_input = temp_string_stream.str();

  std::string c_tminus1 = "Sum(IfDefined(Offset(c1_t, -1)), IfDefined(Offset( c2_t, -1)))";

  // i_t
  os << "component-node name=i1 component=Wi-xr input=Append("
     << spliced_input << ", IfDefined(Offset(r_t, -1)))\n";
  os << "component-node name=i2 component=Wic "
     << " input=" << c_tminus1 << std::endl;
  os << "component-node name=i_t component=i input=Sum(i1, i2)\n";

  // f_t
  os << "component-node name=f1 component=Wf-xr input=Append("
     << spliced_input << ", IfDefined(Offset(r_t, -1)))\n";
  os << "component-node name=f2 component=Wfc "
     << " input=" << c_tminus1 << std::endl;
  os << "component-node name=f_t component=f input=Sum(f1, f2)\n";

  // o_t
  os << "component-node name=o1 component=Wo-xr input=Append("
     << spliced_input << ", IfDefined(Offset(r_t, -1)))\n";
  os << "component-node name=o2 component=Woc input=Sum(c1_t, c2_t)\n";
  os << "component-node name=o_t component=o input=Sum(o1, o2)\n";

  // h_t
  os << "component-node name=h_t component=h input=Sum(c1_t, c2_t)\n";

  // g_t
  os << "component-node name=g1 component=Wc-xr input=Append("
     << spliced_input << ", IfDefined(Offset(r_t, -1)))\n";
  os << "component-node name=g_t component=g input=g1\n";

  // parts of c_t
  os << "component-node name=c1_t component=c1 "
     << " input=Append(f_t, " << c_tminus1 << ")\n";
  os << "component-node name=c2_t component=c2 input=Append(i_t, g_t)\n";

  // m_t
  os << "component-node name=m_t component=m input=Append(o_t, h_t)\n";

  // r_t and p_t
  os << "component-node name=rp_t component=W-m input=m_t\n";
  // Splitting outputs of Wy- node
  os << "dim-range-node name=r_t input-node=rp_t dim-offset=0 "
     << "dim=" << projection_dim << std::endl;

  // y_t
  os << "component-node name=y_t component=Wy- input=rp_t\n";

  // Final affine transform
  os << "component-node name=final_affine component=final_affine input=y_t\n";
  os << "component-node name=posteriors component=logsoftmax input=final_affine\n";
  os << "output-node name=output input=posteriors\n";
  configs->push_back(os.str());
}

// This is a different LSTM config where computation is bunched according
// to inputs this is not complete, it is left here for future comparisons
void GenerateConfigSequenceLstmType2(
    const NnetGenerationOptions &opts,
    std::vector<std::string> *configs) {
  KALDI_ERR << "Not Implemented";
  std::ostringstream os;

  std::vector<int32> splice_context;
  for (int32 i = -5; i < 4; i++)
    if (Rand() % 3 == 0)
      splice_context.push_back(i);
  if (splice_context.empty())
    splice_context.push_back(0);

  int32 input_dim = 10 + Rand() % 20,
      spliced_dim = input_dim * splice_context.size(),
      output_dim = 100 + Rand() % 200,
      cell_dim = 40 + Rand() % 50,
      projection_dim = std::ceil(cell_dim / (Rand() % 10 + 2));

  os << "input-node name=input dim=" << input_dim << std::endl;
  // Parameter Definitions W*(* replaced by - to have valid names)
  os << "component name=W-x type=NaturalGradientAffineComponent input-dim="
     << spliced_dim << " output-dim=" << 4 * cell_dim << std::endl;
  os << "component name=W-r type=NaturalGradientAffineComponent input-dim="
     << projection_dim << " output-dim=" << 4 * cell_dim << std::endl;
  os << "component name=W-m type=NaturalGradientAffineComponent input-dim="
     << cell_dim << " output-dim=" << 2 * projection_dim  << std::endl;
  os << "component name=Wyr type=NaturalGradientAffineComponent input-dim="
     << projection_dim << " output-dim=" << cell_dim << std::endl;
  os << "component name=Wyp type=NaturalGradientAffineComponent input-dim="
     << projection_dim << " output-dim=" << cell_dim << std::endl;
  // Defining the diagonal matrices
  os << "component name=Wic type=PerElementScaleComponent "
     << " dim=" << cell_dim << std::endl;
  os << "component name=Wfc type=PerElementScaleComponent "
     << " dim=" << cell_dim << std::endl;
  os << "component name=Woc type=PerElementScaleComponent "
     << " dim=" << cell_dim << std::endl;
  // Defining the final affine transform
  os << "component name=final_affine type=NaturalGradientAffineComponent "
     << "input-dim=" << cell_dim << " output-dim=" << output_dim << std::endl;
  os << "component name=logsoftmax type=LogSoftmaxComponent dim="
     << output_dim << std::endl;

  // Defining the non-linearities
  //  declare a no-op component so that we can use a sum descriptor's output
  //  multiple times, and to make the config more readable given the equations
  os << "component name=c_t type=NoOpComponent dim="
     << cell_dim << std::endl;
  os << "component name=i_t type=SigmoidComponent dim="
     << cell_dim << std::endl;
  os << "component name=f_t type=SigmoidComponent dim="
     << cell_dim << std::endl;
  os << "component name=o_t type=SigmoidComponent dim="
     << cell_dim << std::endl;
  os << "component name=g type=TanhComponent dim="
     << cell_dim << std::endl;
  os << "component name=h type=TanhComponent dim="
     << cell_dim << std::endl;
  os << "component name=f_t-c_tminus1 type=ElementwiseProductComponent "
     << " input-dim=" << 2 * cell_dim
     << " output-dim=" << cell_dim << std::endl;
  os << "component name=i_t-g type=ElementwiseProductComponent "
     << " input-dim=" << 2 * cell_dim
     << " output-dim=" << cell_dim << std::endl;
  os << "component name=m_t type=ElementwiseProductComponent "
     << " input-dim=" << 2 * cell_dim
     << " output-dim=" << cell_dim << std::endl;


  // Defining the computations
  os << "component-node name=W-x component=W-x input=Append(";
  for (size_t i = 0; i < splice_context.size(); i++) {
    int32 offset = splice_context[i];
    os << "Offset(input, " << offset << ")";
    if (i + 1 < splice_context.size())
      os << ", ";
  }
  os << ")\n";

  os << "component-node name=W-r component=W-r input=IfDefined(Offset(r_t, -1))\n";
  os << "component-node name=W-m component=W-m input=m_t \n";
  os << "component-node name=Wic component=Wic input=IfDefined(Offset(c_t, -1))\n";
  os << "component-node name=Wfc component=Wfc input=IfDefined(Offset(c_t, -1))\n";
  os << "component-node name=Woc component=Woc input=c_t\n";

  // Splitting the outputs of W*m node
  os << "dim-range-node name=r_t input-node=W-m dim-offset=0 "
     << "dim=" << projection_dim << std::endl;
  os << "dim-range-node name=p_t input-node=W-m dim-offset=" << projection_dim
     << " dim=" << projection_dim << std::endl;

  // Splitting outputs of W*x node
  os << "dim-range-node name=W_ix-x_t input-node=W-x dim-offset=0 "
     << "dim=" << cell_dim << std::endl;
  os << "dim-range-node name=W_fx-x_t input-node=W-x "
     << "dim-offset=" << cell_dim << " dim="<<cell_dim << std::endl;
  os << "dim-range-node name=W_cx-x_t input-node=W-x "
     << "dim-offset=" << 2 * cell_dim << " dim="<<cell_dim << std::endl;
  os << "dim-range-node name=W_ox-x_t input-node=W-x "
     << "dim-offset=" << 3 * cell_dim << " dim="<<cell_dim << std::endl;

  // Splitting outputs of W*r node
  os << "dim-range-node name=W_ir-r_tminus1 input-node=W-r dim-offset=0 "
     << "dim=" << cell_dim << std::endl;
  os << "dim-range-node name=W_fr-r_tminus1 input-node=W-r "
     << "dim-offset=" << cell_dim << " dim="<<cell_dim << std::endl;
  os << "dim-range-node name=W_cr-r_tminus1 input-node=W-r "
     << "dim-offset=" << 2 * cell_dim << " dim="<<cell_dim << std::endl;
  os << "dim-range-node name=W_or-r_tminus1 input-node=W-r "
     << "dim-offset=" << 3 * cell_dim << " dim="<<cell_dim << std::endl;

  // Non-linear operations
  os << "component-node name=c_t component=c_t input=Sum(f_t-c_tminus1, i_t-g)\n";
  os << "component-node name=h component=h input=c_t\n";
  os << "component-node name=i_t component=i_t input=Sum(W_ix-x_t, Sum(W_ir-r_tminus1, Wic))\n";
  os << "component-node name=f_t component=f_t input=Sum(W_fx-x_t, Sum(W_fr-r_tminus1, Wfc))\n";
  os << "component-node name=o_t component=o_t input=Sum(W_ox-x_t, Sum(W_or-r_tminus1, Woc))\n";
  os << "component-node name=f_t-c_tminus1 component=f_t-c_tminus1 input=Append(f_t, Offset(c_t, -1))\n";
  os << "component-node name=i_t-g component=i_t-g input=Append(i_t, g)\n";
  os << "component-node name=m_t component=m_t input=Append(o_t, h)\n";

  os << "component-node name=g component=g input=Sum(W_cx-x_t, W_cr-r_tminus1)\n";

  // Final affine transform
  os << "component-node name=Wyr component=Wyr input=r_t\n";
  os << "component-node name=Wyp component=Wyp input=p_t\n";

  os << "component-node name=final_affine component=final_affine input=Sum(Wyr, Wyp)\n";

  os << "component-node name=posteriors component=logsoftmax input=final_affine\n";
  os << "output-node name=output input=posteriors\n";

  configs->push_back(os.str());
}

void GenerateConfigSequence(
    const NnetGenerationOptions &opts,
    std::vector<std::string> *configs) {
start:
  int32 network_type = RandInt(0, 6);
  switch(network_type) {
    case 0:
      GenerateConfigSequenceSimplest(opts, configs);
      break;
    case 1:
      if (!opts.allow_context)
        goto start;
      GenerateConfigSequenceSimpleContext(opts, configs);
      break;
    case 2:
      if (!opts.allow_context || !opts.allow_nonlinearity)
        goto start;
      GenerateConfigSequenceSimple(opts, configs);
      break;
    case 3:
      if (!opts.allow_recursion || !opts.allow_context ||
          !opts.allow_nonlinearity)
        goto start;
      GenerateConfigSequenceRnn(opts, configs);
      break;
    case 4:
      if (!opts.allow_recursion || !opts.allow_context ||
          !opts.allow_nonlinearity)
        goto start;
      GenerateConfigSequenceRnnClockwork(opts, configs);
      break;
    case 5:
      if (!opts.allow_recursion || !opts.allow_context ||
          !opts.allow_nonlinearity)
        goto start;
      GenerateConfigSequenceLstm(opts, configs);
      break;
    case 6:
      if (!opts.allow_recursion || !opts.allow_context ||
          !opts.allow_nonlinearity)
        goto start;
      GenerateConfigSequenceLstm(opts, configs);
      break;
    default:
      KALDI_ERR << "Error generating config sequence.";
  }
}

void ComputeExampleComputationRequestSimple(
    const Nnet &nnet,
    ComputationRequest *request,
    std::vector<Matrix<BaseFloat> > *inputs) {
  KALDI_ASSERT(IsSimpleNnet(nnet));

  int32 left_context, right_context;
  ComputeSimpleNnetContext(nnet, &left_context, &right_context);

  int32 num_output_frames = 1 + Rand() % 10,
      output_start_frame = Rand() % 10,
      num_examples = 1 + Rand() % 10,
      output_end_frame = output_start_frame + num_output_frames,
      input_start_frame = output_start_frame - left_context - (Rand() % 3),
      input_end_frame = output_end_frame + right_context + (Rand() % 3),
      n_offset = Rand() % 2;
  bool need_deriv = (Rand() % 2 == 0);

  request->inputs.clear();
  request->outputs.clear();
  inputs->clear();

  std::vector<Index> input_indexes, ivector_indexes, output_indexes;
  for (int32 n = n_offset; n < n_offset + num_examples; n++) {
    for (int32 t = input_start_frame; t < input_end_frame; t++)
      input_indexes.push_back(Index(n, t, 0));
    for (int32 t = output_start_frame; t < output_end_frame; t++)
      output_indexes.push_back(Index(n, t, 0));
    ivector_indexes.push_back(Index(n, 0, 0));
  }
  request->outputs.push_back(IoSpecification("output", output_indexes));
  if (need_deriv || (Rand() % 3 == 0))
    request->outputs.back().has_deriv = true;
  request->inputs.push_back(IoSpecification("input", input_indexes));
  if (need_deriv && (Rand() % 2 == 0))
    request->inputs.back().has_deriv = true;
  int32 input_dim = nnet.InputDim("input");
  KALDI_ASSERT(input_dim > 0);
  inputs->push_back(
      Matrix<BaseFloat>((input_end_frame - input_start_frame) * num_examples,
                        input_dim));
  inputs->back().SetRandn();
  int32 ivector_dim = nnet.InputDim("ivector");  // may not exist.
  if (ivector_dim != -1) {
    request->inputs.push_back(IoSpecification("ivector", ivector_indexes));
    inputs->push_back(Matrix<BaseFloat>(num_examples, ivector_dim));
    inputs->back().SetRandn();
    if (need_deriv && (Rand() % 2 == 0))
      request->inputs.back().has_deriv = true;
  }
  if (Rand() % 2 == 0)
    request->need_model_derivative = need_deriv;
  if (Rand() % 2 == 0)
    request->store_component_stats = true;
}


static void GenerateRandomComponentConfig(std::string *component_type,
                                          std::string *config) {
  int32 n = RandInt(0, 16);
  std::ostringstream os;
  switch(n) {
    case 0: {
      *component_type = "PnormComponent";
      int32 output_dim = RandInt(1, 50), group_size = RandInt(1, 15),
             input_dim = output_dim * group_size;
      os << "input-dim=" << input_dim << " output-dim=" << output_dim;
      break;
    }
    case 1: {
      *component_type = "NormalizeComponent";
      os << "dim=" << RandInt(1, 50);
      break;
    }
    case 2: {
      *component_type = "SigmoidComponent";
      os << "dim=" << RandInt(1, 50);
      break;
    }
    case 3: {
      *component_type = "TanhComponent";
      os << "dim=" << RandInt(1, 50);
      break;
    }
    case 4: {
      *component_type = "RectifiedLinearComponent";
      os << "dim=" << RandInt(1, 50);
      break;
    }
    case 5: {
      *component_type = "SoftmaxComponent";
      os << "dim=" << RandInt(1, 50);
      break;
    }
    case 6: {
      *component_type = "LogSoftmaxComponent";
      os << "dim=" << RandInt(1, 50);
      break;
    }
    case 7: {
      *component_type = "NoOpComponent";
      os << "dim=" << RandInt(1, 50);
      break;
    }
    case 8: {
      *component_type = "FixedAffineComponent";
      int32 input_dim = RandInt(1, 50), output_dim = RandInt(1, 50);
      os << "input-dim=" << input_dim << " output-dim=" << output_dim;
      break;
    }
    case 9: {
      *component_type = "AffineComponent";
      int32 input_dim = RandInt(1, 50), output_dim = RandInt(1, 50);
      os << "input-dim=" << input_dim << " output-dim=" << output_dim;
      break;
    }
    case 10: {
      *component_type = "NaturalGradientAffineComponent";
      int32 input_dim = RandInt(1, 50), output_dim = RandInt(1, 50);
      os << "input-dim=" << input_dim << " output-dim=" << output_dim;
      break;
    }
    case 11: {
      *component_type = "SumGroupComponent";
      std::vector<int32> sizes;
      int32 num_groups = RandInt(1, 50);
      os << "sizes=";
      for (int32 i = 0; i < num_groups; i++) {
        os << RandInt(1, 5);
        if (i + 1 < num_groups)
          os << ',';
      }
      break;
    }
    case 12: {
      *component_type = "FixedScaleComponent";
      os << "dim=" << RandInt(1, 100);
      break;
    }
    case 13: {
      *component_type = "FixedBiasComponent";
      os << "dim=" << RandInt(1, 100);
      break;
    }
    case 14: {
      *component_type = "NaturalGradientPerElementScaleComponent";
      os << "dim=" << RandInt(1, 100);
      break;
    }
    case 15: {
      *component_type = "PerElementScaleComponent";
      os << "dim=" << RandInt(1, 100);
      break;
    }
    case 16: {
      *component_type = "ElementwiseProductComponent";
      int32 output_dim = RandInt(1, 100), multiple = RandInt(2, 4),
          input_dim = output_dim * multiple;
      os << "input-dim=" << input_dim << " output-dim=" << output_dim;
      break;
    }
    default:
      KALDI_ERR << "Error generating random component";
  }
  *config = os.str();
}

/// Generates random simple component for testing.
Component *GenerateRandomSimpleComponent() {
  std::string component_type, config;
  GenerateRandomComponentConfig(&component_type, &config);
  ConfigLine config_line;
  if (!config_line.ParseLine(config))
    KALDI_ERR << "Bad config line " << config;

  Component *c = Component::NewComponentOfType(component_type);
  if (c == NULL)
    KALDI_ERR << "Invalid component type " << component_type;
  c->InitFromConfig(&config_line);
  return c;
}

bool NnetParametersAreIdentical(const Nnet &nnet1,
                                const Nnet &nnet2,
                                BaseFloat threshold = 1.0e-05) {
  KALDI_ASSERT(nnet1.NumComponents() == nnet2.NumComponents());
  int32 num_components = nnet1.NumComponents();
  for (int32 c = 0; c < num_components; c++) {
    const Component *c1 = nnet1.GetComponent(c),
                    *c2 = nnet2.GetComponent(c);
    KALDI_ASSERT(c1->Type() == c2->Type());
    if (c1->Properties() & kUpdatableComponent) {
      const UpdatableComponent *u1 = dynamic_cast<const UpdatableComponent*>(c1),
                               *u2 = dynamic_cast<const UpdatableComponent*>(c2);
      KALDI_ASSERT(u1 != NULL && u2 != NULL);
      BaseFloat prod11 = u1->DotProduct(*u1), prod12 = u1->DotProduct(*u2),
                prod21 = u2->DotProduct(*u1), prod22 = u2->DotProduct(*u2);
      BaseFloat max_prod = std::max(std::max(prod11, prod12),
                                    std::max(prod21, prod22)),
                min_prod = std::min(std::min(prod11, prod12),
                                    std::min(prod21, prod22));
      if (max_prod - min_prod > threshold * max_prod) {
        KALDI_WARN << "Component '" << nnet1.GetComponentName(c)
                   << "' differs in nnet1 versus nnet2: prod(11,12,21,22) = "
                   << prod11 << ',' << prod12 << ',' << prod21 << ',' << prod22;
        return false;
      }
    }
  }
  return true;
}

void GenerateSimpleNnetTrainingExample(
    int32 num_supervised_frames,
    int32 left_context,
    int32 right_context,
    int32 output_dim,
    int32 input_dim,
    int32 ivector_dim,
    NnetExample *example) {
  KALDI_ASSERT(num_supervised_frames > 0 && left_context >= 0 &&
               right_context >= 0 && output_dim > 0 && input_dim > 0
               && example != NULL);
  example->io.clear();

  int32 feature_t_begin = RandInt(0, 2);
  int32 num_feat_frames = left_context + right_context + num_supervised_frames;
  Matrix<BaseFloat> input_mat(num_feat_frames, input_dim);
  input_mat.SetRandn();
  NnetIo input_feat("input", feature_t_begin, input_mat);
  if (RandInt(0, 1) == 0)
    input_feat.features.Compress();
  example->io.push_back(input_feat);

  if (ivector_dim > 0) {
    // Create a feature for the iVectors.  iVectors always have t=0 in the
    // current framework.
    Matrix<BaseFloat> ivector_mat(1, ivector_dim);
    ivector_mat.SetRandn();
    NnetIo ivector_feat("ivector", 0, ivector_mat);
    if (RandInt(0, 1) == 0)
      ivector_feat.features.Compress();
    example->io.push_back(ivector_feat);
  }

  {  // set up the output supervision.
    Posterior labels(num_supervised_frames);
    for (int32 t = 0; t < num_supervised_frames; t++) {
      int32 num_labels = RandInt(1, 3);
      BaseFloat remaining_prob_mass = 1.0;
      for (int32 i = 0; i < num_labels; i++) {
        BaseFloat this_prob = (i+1 == num_labels ? 1.0 : RandUniform()) *
            remaining_prob_mass;
        remaining_prob_mass -= this_prob;
        labels[t].push_back(std::pair<int32, BaseFloat>(RandInt(0, output_dim-1),
                                                        this_prob));
      }
    }
    int32 supervision_t_begin = feature_t_begin + left_context;
    NnetIo output_sup("output", output_dim, supervision_t_begin,
                      labels);
    example->io.push_back(output_sup);
  }
}

bool ExampleApproxEqual(const NnetExample &eg1,
                        const NnetExample &eg2,
                        BaseFloat delta) {
  if (eg1.io.size() != eg2.io.size())
    return false;
  for (size_t i = 0; i < eg1.io.size(); i++) {
    NnetIo io1 = eg1.io[i], io2 = eg2.io[i];
    if (io1.name != io2.name || io1.indexes != io2.indexes)
      return false;
    Matrix<BaseFloat> feat1, feat2;
    io1.features.GetMatrix(&feat1);
    io2.features.GetMatrix(&feat2);
    if (!ApproxEqual(feat1, feat2, delta))
      return false;
  }
  return true;
}


} // namespace nnet3
} // namespace kaldi
