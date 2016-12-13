#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat train_pos_cleaned.txt train_neg_cleaned.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > vocab_cleaned.txt
