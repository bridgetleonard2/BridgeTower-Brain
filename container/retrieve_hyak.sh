# After running encoding_models.py in Hyak, retrieve the results to local repo
# Usage: sh retrieve_hyak.sh

# Retrieve movie_to_story results
scp -r bll313@klone.hyak.uw.edu:/mmfs1/gscratch/scrubbed/bll313/results/movie_to_story/S1 "C:/Users/Bridget Leonard/Desktop/BridgeTower-Brain/results/movie_to_story"

# Retrieve story_to_movie results
scp -r bll313@klone.hyak.uw.edu:/mmfs1/gscratch/scrubbed/bll313/results/story_to_movie/S1 "C:/Users/Bridget Leonard/Desktop/BridgeTower-Brain/results/story_to_movie"

# Retrieve within vision model results
scp -r bll313@klone.hyak.uw.edu:/mmfs1/gscratch/scrubbed/bll313/results/vision_model/S1 "C:/Users/Bridget Leonard/Desktop/BridgeTower-Brain/results/vision_only"

# Retrieve face and landscape results
scp -r bll313@klone.hyak.uw.edu:/mmfs1/gscratch/scrubbed/bll313/results/faces/* "C:/Users/Bridget Leonard/Desktop/BridgeTower-Brain/results/face_check"
scp -r bll313@klone.hyak.uw.edu:/mmfs1/gscratch/scrubbed/bll313/results/landscapes/* "C:/Users/Bridget Leonard/Desktop/BridgeTower-Brain/results/landscape_check"