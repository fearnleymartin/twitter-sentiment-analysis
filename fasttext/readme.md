# How to run

Using the provided pre-formatted files, you can simply do:
```python ../../..vagrant/fasttext/run.py```
You will need to post-process the file afterwards to get a file suitable for kaggle submissions

Note: we are running fasttext on a virtual machine on windows using vagrant. The filepaths will vary for other setups.

The pre-processing and post-processing are explained in more detail in the following paragraphs.
To run with other data input files, you will need to change the input/output paths.

# Preprocessing

We first tokenise the text, then we need to format the text for usage with FastText

## tokenisation
```python preprocessing/preprocessing.py```
Applies nltk tweet tokenizer to tweets.
Run this file to generate tokenised tweet files. Make sure to specify correct input/output files

## FastText formatting
```python fasttext/pre_formatting.py```
Adds labels in correct format for fasttext and combines pos/neg into single file. Make sure to specify correct input/output filepaths

# Fasttext 
To run fasttext: enter fasttext directory
`cd fastText`

To generate model (can change input and output as needed. Stuff you see after the /vagrant/ is the root directory)
`./fasttext supervised -input ../../../vagrant/train_fasttext_full.txt -output model_full`

To evaluate on text :
`./fasttext predict model.bin ../../../vagrant/test_data_processed.txt > ../../../vagrant/results_fasttext_full.txt`

# FastText postprocessing

```python fasttext/postprocessing.py ``` 
Adapts the fasttext output to kaggle submission format

Specify input/outputs

# Cross Validation

We have implemented a cross validation setup for fasttext. 
```
python ../../../vagrant/fasttext/cross_validation.py
```

You can specify the input file to use and the number of folds.  
This will run a cross validation and calculate the average accuracy of the model.

# Install a ubuntu virtual machine with vagrant (windows only)

For windows users only, as fastText only runs on unix systems. The following step will help you set up a ubuntu machine and build fasttext

Download and install virtual box and vagrant

In python settings, go to vagrant tab and select the exectuable file for Vagrant.  
In my case it is: C:/HashiCorp/Vagrant/bin/vagrant.exe  
Still in the settings, add a new box (ubuntu/trusty64 enter this as box name and box url).  
This will download a ubuntu image (can take a while, file size about 300 M)  

In menu bar in pycharm: tools/vagrant/up  
This will launch the virtual machine and install the necessary dependencies (including fasttext)

To enter virtual machine: select from pycharm menu bar  
Tools/start ssh session  
This will open a command line interface which allows you to work in the virtual machine.  
Your files from pycharm should be synchronised.  
(to see them `cd ../../vagrant` then `ls` )