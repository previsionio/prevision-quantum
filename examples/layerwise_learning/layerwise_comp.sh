#!/bin/bash

run_spirals() {
  echo starting input file: $1

  date=$(date +"%I:%M:%S")

  if [ -d $1 ]
  then
    # if directory exists, do not run the calculation
    echo "$date directory already exists, not overwriting for safety" >> log.txt
  else
    # if directory does not exist yet, run the calculation
    echo "$date starting input file $1, with nb_turns $2" >> log.txt
    python3 $1 1.0
  fi
}

export -f run_spirals

parallel run_spirals ::: global_training.py layerwise_training.py global_training_id_block.py layerwise_training_id_block.py