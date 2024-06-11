#!/bin/bash

total_iterations=$1
regime=$2
offset=${3:-0}

for ((counter=offset+1; counter<=total_iterations+offset; counter++)); do
  echo "Queueing Surrogate Job $((counter-offset))/$total_iterations"
  sbatch final_evaluate.sh "$counter" "$regime"
  sleep 0.2
done

