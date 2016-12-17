#!/usr/bin/env bash
sudo apt-get install git
sudo apt-get install python-numpy python-scipy
pip install cython
git clone https://github.com/facebookresearch/fastText.git
cd fastText
make