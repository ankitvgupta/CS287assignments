# Secondary Structure Prediction with Structured Prediction (and Bidirectional LSTMs)

A final project for Harvard's [CS287](http://cs287.fas.harvard.edu) by @ankitvgupta and @tomsilver

## Overview

This project contributes to [the literature](http://www.esciencecentral.org/journals/a-short-review-of-deep-learning-neural-networks-in-protein-structureprediction-problems-2379-1764-1000139.php?aid=64286) on deep learning approaches to protein Secondary Structure Prediction.

Preprocessing, including parsing CB513, CullPDB, and HUMAN and filtering the latter, is performed in Python. Models are built (in `models.lua`), trained, and evaluated (in `modelrunner.lua`) in Torch. Larger experiments are run on GPUs using the appropriate Torch modules.

## Quick Start

1. Download and unzip the [CB513 and filtered CullPDB datasets](http://princeton.edu/~jzthree/datasets/ICML2014/) into a `data` directory.
2. Run `python preprocessing/preprocess.py EPRINC --dwin=1` to preprocess the CullPDB and CB513 datasets.
3. Run `th finalproject.lua -datafile EPRINC_1.hdf5 -classifier rnn -b 128 -additional_features -bidirectional -memm_layer` to train and test an Bi-LSTM-MEMM with default parameters.

## Preprocessing

### Preprocessing datasets

    usage: preprocess.py [-h] [--test [TEST]] [--dwin [DWIN]] [train_dataset]
    
The primary datasets of interest are the CullPDB training set and the CB513 test set. To parse these together with a window size of 3, run

    python preprocessing/preprocess.py EPRINC --dwin 3
    
To preprocess CullPDB *sequence-only* and split it into train and test sets, run 

    python preprocessing/preprocess.py PRINC
    
To preprocess CullPDB (as training) and CB513 (as test) both *sequence-only*, run

    python preprocessing/preprocess.py PRINC --test CB513
    
To preprocess the full unfiltered HUMAN dataset and split it into train and test sets, run 

    python preprocessing/preprocess.py HUMAN
    
To preprocess the filtered HUMAN dataset (see below) and split it into train and test sets, run 

    python preprocessing/preprocess.py FILT  

### Filtering HUMAN

The HUMAN dataset can be accessed at [http://www.rcsb.org/pdb/files/ss.txt](http://www.rcsb.org/pdb/files/ss.txt). Download and store in the same `data` directory as the CB513 data (this is required).

To filter out sequences that are homologous with ones in CB513, use

    preprocessing/princfilter.py [-h] cb513 train start count

    positional arguments:
      cb513       CB513 path
      train       Training set path
      start       Start index
      count       Number of indices

Then edit and use `preprocessing/createfilteredseqs.py` to create the final sequences before preprocessing.

## Building and Evaluating Models

### MEMMs
TODO

### LSTMs
TODO



