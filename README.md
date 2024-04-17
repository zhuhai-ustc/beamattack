# beamattack@PAKDD2023


## Requirements

- boto3==1.26.28
- botocore==1.29.28
- torch == 1.12.1+cu116
- tensorflow-gpu == 2.11.0
- tensorflow-hub == 0.12.0
- numpy == 1.23.2
- nltk == 3.7
- scipy == 1.9.1
- pattern==3.6



## Datesets and Victim Model
There are  MR, SST-2 , AG, Yahoo and  SNLI, MNLI and MNLIm datasets. 
We adopt the pretrained models provided including BERT,CNN,LSTM. 
These data and models are adopted from   [TextFooler](https://github.com/jind11/TextFooler).


## Dependencies
glove.6B.200d.txt and  counter-fitted-vectors.txt  
can be obtained from  [TextFooler](https://github.com/jind11/TextFooler)


## File Description
- beamAttack_classification.py: Attack the victim model for text classification with beamAttack.
- beamAttack_nli.py: Attack the victim model for textual entailment with beamAttack.

## run
bash run_mr.sh
