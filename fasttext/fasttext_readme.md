# Install the virtual machine with vagrant

Download and install virtual box and vagrant

In python settings, go to vagrant tab and select the exectuable file for vagrant  
in my case it is: C:/HashiCorp/Vagrant/bin/vagrant.exe  
Still in the settings, add a new box (ubuntu/trusty64 enter this as box name and box url)  
This will download a ubuntu image (can take a while about 300 M)  

In menu bar in pycharm: tools/vagrant/up  
This will launch the virtual machine and install the necessary dependencies (including fasttext)

To enter virtual machine:  
Tools/start ssh session  
This will open a command line interface which allows you to work in the virtual machine.  
Your files from pycharm should be synchronised.  
(to see them `cd ../../vagrant` then `ls` )

# Preprocessing

## tokenisation
In file preprocessing/preprocessing.py  
Applies nltk tweet tokenizer to tweets

Run this file to generate tokenised tweet files. Make sure correct lines are commented/decommented to run on correct files.

## fasttext formatting
In file fasttext.py  
Adds labels in correct format for fasttext

Run to generate input file for fasttext. Again make sure correct lines are commented/decommented. For preprocessing, you only want to decomment preprocess lines, make sure post processing lines are commented.

# Fasttext 
To run fasttext: enter fasttext directory
`cd fasttext`

To generate model (can change input and output as needed. Stuff you see after the /vagrant/ is the root directory in pycharm)
`./fasttext supervised -input ../../../vagrant/train_fasttext_full.txt -output model_full`

To evaluate on text :
`./fasttext predict model.bin ../../../vagrant/test_data_processed.txt > ../../../vagrant/results_fasttext_full.txt`

# fasttext postprocessing

In file fasttext.py  
Adapts the fasttext output to kaggle submission format

Run the file with only postprocessing lines decommented
