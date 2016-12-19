# Install the virtual machine with vagrant (windows only)

Download and install virtual box and vagrant

In python settings, go to vagrant tab and select the exectuable file for vagrant  
in my case it is: C:/HashiCorp/Vagrant/bin/vagrant.exe  
Still in the settings, add a new box (ubuntu/trusty64 enter this as box name and box url)  
This will download a ubuntu image (can take a while about 300 M)  

In menu bar in pycharm: tools/vagrant/up  
This will launch the virtual machine and install the necessary dependencies (including fasttext)

To enter virtual machine: select from pycharm menubar  
Tools/start ssh session  
This will open a command line interface which allows you to work in the virtual machine.  
Your files from pycharm should be synchronised.  
(to see them `cd ../../vagrant` then `ls` )

# Preprocessing

We first tokenise the text, then we need to format the text for usage with fasttext

## tokenisation
In file preprocessing/preprocessing.py  
Applies nltk tweet tokenizer to tweets

Run this file to generate tokenised tweet files. Make sure to specify correct input/output files

## fasttext formatting
In file fasttext/pre_formatting.py  
Adds labels in correct format for fasttext and combines pos/neg into single file

Make sure to specify correct input/output filepaths

# Fasttext 
To run fasttext: enter fasttext directory
`cd fasttext`

To generate model (can change input and output as needed. Stuff you see after the /vagrant/ is the root directory in pycharm)
`./fasttext supervised -input ../../../vagrant/train_fasttext_full.txt -output model_full`

To evaluate on text :
`./fasttext predict model.bin ../../../vagrant/test_data_processed.txt > ../../../vagrant/results_fasttext_full.txt`

# fasttext postprocessing

In file fasttext/postprocessing.py  
Adapts the fasttext output to kaggle submission format

Specify input/outputs

# cross validation

```
python ../../../vagrant/fasttext/cross_validation.py
```

You can specify the input file to use and the number of folds.  
This will run a cross validation and calculate the average accuracy of the model.