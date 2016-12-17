#!/usr/bin/env bash
yes | sudo apt-get install git
yes | sudo apt-get install python-numpy python-scipy
pip install cython
git clone https://github.com/facebookresearch/fastText.git
cd fastText
make