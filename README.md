# scientific-re
CNN for Relation Classification.

The repository contains sample data for reproducing one experiment from the correspondent paper:
  - model: RC using SciBERT embeddings
  - train: 2A w/o CR
  - test: AI-ML
To run the experiment simply run python3 main.py

The complete data setups will be released upon accetance.

All the parameter are in src/parameters/parameters.py

The sample data is in sample-data in the following format:
instance-id \t #-sentence \t #-token \t token \t token-type{entity-id,"NO-ENTITY"}
