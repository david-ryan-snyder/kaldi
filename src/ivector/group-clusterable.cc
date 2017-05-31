// ivector/group-clusterable.cc

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


#include "ivector/group-clusterable.h"

namespace kaldi {

BaseFloat GroupClusterable::Objf() const {
  // TODO (current only using Distance())
  KALDI_ERR << "GroupClusterable::Objf isn't enabled yet"
            << "  Use Distance() instead.";
  return total_distance_;
}

void GroupClusterable::SetZero() {
  points_.clear();
  total_distance_ = 0;
}

void GroupClusterable::Add(const Clusterable &other_in) {
  const GroupClusterable *other =
      static_cast<const GroupClusterable*>(&other_in);
  points_.insert(other->points_.begin(), other->points_.end());
}

void GroupClusterable::Sub(const Clusterable &other_in) {
  const GroupClusterable *other =
      static_cast<const GroupClusterable*>(&other_in);
  for (std::set<int32>::const_iterator itr_i = other->points_.begin();
       itr_i != other->points_.end(); ++itr_i) {
    points_.erase(*itr_i);
  }
}

BaseFloat GroupClusterable::Normalizer() const {
  return points_.size();
}

Clusterable *GroupClusterable::Copy() const {
  GroupClusterable *ans = new GroupClusterable(points_, scores_, max_distance_);
  return ans;
}

void GroupClusterable::Scale(BaseFloat f) {
  // TODO, doesn't do anything yet
  KALDI_ERR << "GroupClusterable::Scale isn't enabled yet";
  return;
}

void GroupClusterable::Write(std::ostream &os, bool binary) const {
  // TODO, doesn't do anything yet
  KALDI_ERR << "GroupClusterable::Write isn't enabled yet";
  return;
}

Clusterable *GroupClusterable::ReadNew(std::istream &is, bool binary) const {
  // TODO
  KALDI_ERR << "GroupClusterable::ReadNew isn't enabled yet";
  return NULL;
}

BaseFloat GroupClusterable::Distance(const Clusterable &other_in) const {
  const GroupClusterable *other =
      static_cast<const GroupClusterable*>(&other_in);
  BaseFloat d = 0.0;
  for (std::set<int32>::iterator itr_i = points_.begin();
    itr_i != points_.end(); ++itr_i) {
    for (std::set<int32>::iterator itr_j = other->points_.begin();
      itr_j != other->points_.end(); ++itr_j) {
      if (scores_->find(std::pair<int32, int32>(*itr_i, *itr_j))
          != scores_->end()) {
        d += (*scores_)[std::pair<int32, int32>(*itr_i, *itr_j)];
      // If there is no entry for i,j, assign it the maximum distance.
      } else {
        d += max_distance_;
      }
    }
  }
  // Average pairwise distance between points in self and other
  return d / (points_.size() * other->points_.size());
}
}
