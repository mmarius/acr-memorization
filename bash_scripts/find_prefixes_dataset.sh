#!/usr/bin/env bash

PROJECT_DIR=/home/nlp/users/mmosba/projects/acr-memorization

# Specify arguments 
DATASET=famous_quotes
N_TOKENS_IN_PROMPT=$1
MAX_TOKENS=$2

# Loop over samples
for data_idx in {0..49}
do
    # Loop over seeds
    for seed in {1..5}
    do
        # Run the script
        python $PROJECT_DIR/prompt_minimization.py dataset=$DATASET data_idx=$data_idx seed=$seed \
            n_tokens_in_prompt=$N_TOKENS_IN_PROMPT \
            max_tokens=$MAX_TOKENS \
            stop_early=false
    done
done

