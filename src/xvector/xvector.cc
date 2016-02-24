// xvector/xvector.cc

// Copyright 2016  David Snyder

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

#include "xvector/xvector.h"

namespace kaldi {

void ComputeXvectorObjfAndDeriv(
    const CuMatrixBase<BaseFloat> &xvector_pairs,
    const CuSpMatrix<BaseFloat> &S,
    BaseFloat b, CuMatrixBase<BaseFloat> *deriv_xvector,
    CuVector<BaseFloat> *deriv_S, BaseFloat *deriv_b,
    CuMatrixBase<BaseFloat> *raw_scores,
    BaseFloat *tot_objf,
    BaseFloat *tot_weight) {

  int32 S_dim = S.NumCols() * (S.NumCols() + 1) / 2,
        N = xvector_pairs.NumRows(),
        xvector_dim = xvector_pairs.NumCols();
  (*tot_objf) = 0;

  if (deriv_xvector == NULL)
    KALDI_ASSERT(deriv_S == NULL && deriv_b == NULL);
  else {
    KALDI_ASSERT(deriv_xvector->NumCols() == xvector_dim);
    KALDI_ASSERT(deriv_xvector->NumRows() == N);
    KALDI_ASSERT(deriv_S->Dim() == S_dim);
  }

  CuMatrix<BaseFloat> S_tmp(S),
                      P(N, xvector_dim),
                      Q(N, N),
                      R(N, N),
                      scores(N, N),                 // The raw scores.
                      objf_terms(N, N, kUndefined),
                      objf_deriv_terms(N, N,        // Derivative of the
                                       kUndefined); // objf w.r.t. the scores.
  CuVector<BaseFloat> r(N);

  P.AddMatMat(1.0, xvector_pairs, kNoTrans, S_tmp, kNoTrans, 0.0);
  r.AddDiagMatMat(1.0, xvector_pairs, kNoTrans, P, kTrans, 0.0);
  R.AddVecToRows(1.0, r);
  Q.SymAddMat2(1.0, xvector_pairs, kNoTrans, 0.0);
  Q.CopyLowerToUpper();
  scores.AddMat(1.0, Q, kNoTrans);
  scores.AddMat(-1.0, R, kTrans);
  scores.AddMat(-1.0, R, kNoTrans);
  scores.Add(b);
  if (raw_scores != NULL)
    raw_scores->CopyFromMat(scores);

  cu::ComputeXvectorObjfFromScores(scores, &objf_terms, &objf_deriv_terms);
  CuVector<BaseFloat> objf_terms_vec(N);
  objf_terms_vec.AddRowSumMat(1.0, objf_terms);
  (*tot_objf) = objf_terms_vec.Sum();

  if (deriv_xvector != NULL) {
    // Some vector and matrix quantities for computing the
    // derivatives.
    CuMatrix<BaseFloat> objf_deriv_terms_trans(objf_deriv_terms, kTrans),
             S_deriv_part(N, xvector_dim),
             S_deriv(xvector_dim, xvector_dim);
    CuVector<BaseFloat> cvec_rows(N),
                        cvec_cols(N);
    cvec_rows.AddRowSumMat(1.0, objf_deriv_terms, 1.0);
    cvec_cols.AddRowSumMat(1.0, objf_deriv_terms_trans, 1.0);
    CuVector<BaseFloat> cvec(cvec_rows);
    cvec.AddVec(1.0, cvec_cols, 1.0);

    // Compute derivative of the objf with respect to the xvectors.
    CuMatrix<BaseFloat> SX(N, xvector_dim);
    SX.AddMatMat(1.0, xvector_pairs, kNoTrans, S_tmp, kNoTrans, 0.0);
    deriv_xvector->AddDiagVecMat(-1.0, cvec_rows, xvector_pairs,
                                 kNoTrans, 0.0);
    deriv_xvector->AddMatMat(-1.0, objf_deriv_terms, kTrans,
                             xvector_pairs, kNoTrans, 1.0);
    deriv_xvector->AddDiagVecMat(2.0, cvec_cols, SX,
                                kNoTrans, 1.0);
    deriv_xvector->AddMatMat(2.0, objf_deriv_terms, kNoTrans,
                             SX, kNoTrans, 1.0);

    // Compute derivative of the objf with respect to the symmetric matrix
    // S.
    S_deriv_part.AddDiagVecMat(2.0, cvec, xvector_pairs,
                              kNoTrans, 0.0);
    S_deriv.AddMatMat(1.0, xvector_pairs, kTrans, S_deriv_part,
                      kNoTrans, 1.0);
    CuSpMatrix<BaseFloat> S_deriv_tmp(S_deriv);
    S_deriv_tmp.ScaleDiag(0.5);
    deriv_S->CopyFromVec(CuSubVector<BaseFloat>(S_deriv_tmp.Data(),
                            S_dim));

    // Compute derivative of objf with respect to the scalar offset b.
    (*deriv_b) = -objf_deriv_terms.Sum();
  }
  (*tot_weight) = N;
}

} // namespace kaldi
