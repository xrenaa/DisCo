#!/bin/bash
# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

echo "Downloading cars dataset."
if [[ ! -d "cars" ]]; then
  wget -O nips2015-analogy-data.tar.gz http://www.scottreed.info/files/nips2015-analogy-data.tar.gz
  tar xzf nips2015-analogy-data.tar.gz
  rm nips2015-analogy-data.tar.gz
  mv data/cars .
  rm -r data
fi
echo "Downloading cars completed!"

echo "Downloading dSprites dataset."
if [[ ! -d "dsprites" ]]; then
  mkdir dsprites
  wget -O dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
fi
echo "Downloading dSprites completed!"

echo "Downloading scream picture."
if [[ ! -d "scream" ]]; then
  mkdir scream
  wget -O scream/scream.jpg https://upload.wikimedia.org/wikipedia/commons/f/f4/The_Scream.jpg
fi
echo "Downloading scream completed!"

echo "Downloading mpi3d_toy dataset."
if [[ ! -d "mpi3d_toy" ]]; then
  mkdir mpi3d_toy
  wget -O mpi3d_toy/mpi3d_toy.npz https://storage.googleapis.com/disentanglement_dataset/data_npz/sim_toy_64x_ordered_without_heldout_factors.npz
fi
echo "Downloading mpi3d_toy completed!"