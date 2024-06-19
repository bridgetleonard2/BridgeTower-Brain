#!/bin/bash

# Get script arguments
subject=$1
layer=$2

if [ -z "$subject" ] || [ -z "$layer" ]; then
    echo "Usage: $0 <subject> <layer>"
    exit 1
fi

# visualize movie_to_story results
python .\container\visualize_correlations.py $subject $layer vision

# visualize story_to_movie results
python .\container\visualize_correlations.py $subject $layer language

# visualize within vision model results
python .\container\visualize_correlations.py $subject $layer visionONLY

# visualize face and landscape results
python .\container\visualize_predictions.py $subject $layer face
python .\container\visualize_predictions.py S1 $layer landscape