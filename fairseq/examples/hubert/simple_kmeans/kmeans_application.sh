#!/bin/bash

# Check if the split argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <split>"
    echo "A split can be \"valid\" or \"train\""
    exit 1
fi

# Define variables
split="$1"
nshard=12 # run in parallel on all cores
max_rank=11 # nshard - 1
feat_dir="../../../../features"
km_path="../../../../valid.km" # kmeans clustering on validation data
lab_dir="../../../../lab"

# Loop through the shards and dispatch the jobs
for rank in $(seq 0 $max_rank); do
    # Run the command in the background
    TORCHAUDIO_USE_FFMPEG=0 TORCHAUDIO_USE_SOX=0 PYTORCH_ENABLE_MPS_FALLBACK=1 python dump_km_label.py $feat_dir $split $km_path $nshard $rank $lab_dir &
done

# Wait for all background jobs to finish
wait

echo "All k means applications completed."

for rank in $(seq 0 $max_rank); do
  cat $lab_dir/${split}_${rank}_${nshard}.km
done > $lab_dir/${split}.km

echo "Merged all shards"
