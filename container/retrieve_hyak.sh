#!/bin/bash
# After running encoding_models.py in Hyak, retrieve the results to local repo
# Usage: sh retrieve_hyak.sh S1 8

# Get script arguments
subject=$1
layer=$2

if [ -z "$subject" ] || [ -z "$layer" ]; then
    echo "Usage: $0 <subject> <layer>"
    exit 1
fi

# Retrieve movie_to_story results
scp -r bll313@klone.hyak.uw.edu:/mmfs1/gscratch/scrubbed/bll313/results/movie_to_story/$subject/layer$layer* "C:/Users/Bridget Leonard/Desktop/BridgeTower-Brain/results/movie_to_story/$subject"

# Retrieve story_to_movie results
scp -r bll313@klone.hyak.uw.edu:/mmfs1/gscratch/scrubbed/bll313/results/story_to_movie/$subject/layer$layer* "C:/Users/Bridget Leonard/Desktop/BridgeTower-Brain/results/story_to_movie/$subject"

# Retrieve within vision model results
scp -r bll313@klone.hyak.uw.edu:/mmfs1/gscratch/scrubbed/bll313/results/vision_model/$subject/layer$layer* "C:/Users/Bridget Leonard/Desktop/BridgeTower-Brain/results/vision_only/$subject"

# Retrieve face and landscape results
scp -r bll313@klone.hyak.uw.edu:/mmfs1/gscratch/scrubbed/bll313/results/faces/$subject/layer$layer* "C:/Users/Bridget Leonard/Desktop/BridgeTower-Brain/results/face_check/$subject"
scp -r bll313@klone.hyak.uw.edu:/mmfs1/gscratch/scrubbed/bll313/results/landscapes/$subject/layer$layer* "C:/Users/Bridget Leonard/Desktop/BridgeTower-Brain/results/landscape_check/$subject"