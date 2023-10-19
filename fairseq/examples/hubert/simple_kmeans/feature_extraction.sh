#!/bin/bash

# Check if the split argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <split>"
    echo "A split can be \"valid\" or \"train\""
    exit 1
fi

# Define variables
metadata_dir="../../../../metadata"
split="$1"
ckpt_path="../../../../hubert_base_ls960.pt"
layer=5 # get the 6th layer out of 12
nshard=12 # run in parallel on all cores
max_rank=11 # nshard - 1
output_dir="../../../../features"

# Loop through the shards and dispatch the jobs
for rank in $(seq 0 $max_rank); do
    # Run the command in the background
    TORCHAUDIO_USE_FFMPEG=0 TORCHAUDIO_USE_SOX=0 PYTORCH_ENABLE_MPS_FALLBACK=1 python dump_hubert_feature.py $metadata_dir $split $ckpt_path $layer $nshard $rank $output_dir &
done

# Wait for all background jobs to finish
wait

echo "All feature extraction jobs completed."
