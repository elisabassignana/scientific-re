# Relation Classification on Scientific Texts
CNN for Relation Classification.

*What Do You Mean by Relation Extraction?* A Survey on Datasets and Study on Scientific Relation Classification [[pdf]](https://aclanthology.org/2022.acl-srw.7/)

The repository contains sample data for reproducing one experiment from the corresponding paper:
  - model: CNN using SciBERT embeddings
  - train: 2A w/o CR
  - test: AI-ML
  
To rerun the experiment run ``python3 main.py``

All the parameter are in ``src/parameters/parameters.py``

The data files follow the format:

>instance-id&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#sentence&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#token&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;token&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;token-type{entity-id,"NO-ENTITY"}

The relation files follow the format:
>entity-1-id&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;entity-2-id&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;relation-type
