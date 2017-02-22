#!/bin/bash

python project1_partitions_preprocess.py data/set_train train 278 25 25 25 > preprocess_train.txt 
python project1_partitions_preprocess.py data/set_test test 138 25 25 25 > preprocess_test.txt

python project1_partitions_submission.py preprocess_train.txt preprocess_test.txt ridge > predictions.csv

