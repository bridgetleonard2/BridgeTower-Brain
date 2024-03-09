import numpy as np
import os
import re
from scipy.signal import resample
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
from bridgetower_functions import prep_data
from bridgetower_functions import calc_correlation
from bridgetower_functions import remove_nan

print("Loading feature data")
caption_to_image_matrices = np.load("results/feature_alignment/caption_to_image_matrices.npy")
image_to_caption_matrices = np.load("results/feature_alignment/image_to_caption_matrices.npy")

# Load feature vectors
# movie data
test = np.load("results/feature_vectors/movie/test_data.npy")
train00 = np.load("results/feature_vectors/movie/train_00_data.npy")
train01 = np.load("results/feature_vectors/movie/train_01_data.npy")
train02 = np.load("results/feature_vectors/movie/train_02_data.npy")
train03 = np.load("results/feature_vectors/movie/train_03_data.npy")
train04 = np.load("results/feature_vectors/movie/train_04_data.npy")
train05 = np.load("results/feature_vectors/movie/train_05_data.npy")
train06 = np.load("results/feature_vectors/movie/train_06_data.npy")
train07 = np.load("results/feature_vectors/movie/train_07_data.npy")
train08 = np.load("results/feature_vectors/movie/train_08_data.npy")
train09 = np.load("results/feature_vectors/movie/train_09_data.npy")
train10 = np.load("results/feature_vectors/movie/train_10_data.npy")
train11 = np.load("results/feature_vectors/movie/train_11_data.npy")

# story data
alternateithicatom = np.load("results/feature_vectors/story/alternateithicatom_data.npy")
avatar = np.load("results/feature_vectors/story/avatar_data.npy")
howtodraw = np.load("results/feature_vectors/story/howtodraw_data.npy")
legacy = np.load("results/feature_vectors/story/legacy_data.npy")
life = np.load("results/feature_vectors/story/life_data.npy")
myfirstdaywiththeyankees = np.load("results/feature_vectors/story/myfirstdaywiththeyankees_data.npy")
naked = np.load("results/feature_vectors/story/naked_data.npy")
odetostepfather = np.load("results/feature_vectors/story/odetostepfather_data.npy")
souls = np.load("results/feature_vectors/story/souls_data.npy")
undertheinfluence = np.load("results/feature_vectors/story/undertheinfluence_data.npy")

test_transformed = np.dot(test, image_to_caption_matrices.T)
train00_transformed = np.dot(train00, image_to_caption_matrices.T)
train01_transformed = np.dot(train01, image_to_caption_matrices.T)
train02_transformed = np.dot(train02, image_to_caption_matrices.T)
train03_transformed = np.dot(train03, image_to_caption_matrices.T)
train04_transformed = np.dot(train04, image_to_caption_matrices.T)
train05_transformed = np.dot(train05, image_to_caption_matrices.T)
train06_transformed = np.dot(train06, image_to_caption_matrices.T)
train07_transformed = np.dot(train07, image_to_caption_matrices.T)
train08_transformed = np.dot(train08, image_to_caption_matrices.T)
train09_transformed = np.dot(train09, image_to_caption_matrices.T)
train10_transformed = np.dot(train10, image_to_caption_matrices.T)
train11_transformed = np.dot(train11, image_to_caption_matrices.T)

alternateithicatom_transformed = np.dot(alternateithicatom, caption_to_image_matrices.T)
avatar_transformed = np.dot(avatar, caption_to_image_matrices.T)
howtodraw_transformed = np.dot(howtodraw, caption_to_image_matrices.T)
legacy_transformed = np.dot(legacy, caption_to_image_matrices.T)
life_transformed = np.dot(life, caption_to_image_matrices.T)
myfirstdaywiththeyankees_transformed = np.dot(myfirstdaywiththeyankees, caption_to_image_matrices.T)
naked_transformed = np.dot(naked, caption_to_image_matrices.T)
odetostepfather_transformed = np.dot(odetostepfather, caption_to_image_matrices.T)
souls_transformed = np.dot(souls, caption_to_image_matrices.T)
undertheinfluence_transformed = np.dot(undertheinfluence, caption_to_image_matrices.T)

print("Loading encoding models")
vision_encoding = np.load("results/encoding_model/movie/" + subject + "_coefficients.npy")
print("Check if vision matrix full of zeroes:", np.all(vision_encoding == 0))

language_encoding = np.load("results/encoding_model/story/" + subject + "_coefficients.npy")
print("Check if language matrix full of zeroes:", np.all(language_encoding == 0))

print("Start movie -> story analysis")
# Load fmri data
print("Loading fMRI data for subject ", subject)

alternateithicatom = np.load("data/fmri_data/storydata/" + subject + "/alternateithicatom.npy")
avatar = np.load("data/fmri_data/storydata/" + subject + "/avatar.npy")
howtodraw = np.load("data/fmri_data/storydata/" + subject + "/howtodraw.npy")
legacy = np.load("data/fmri_data/storydata/" + subject + "/legacy.npy")
life = np.load("data/fmri_data/storydata/" + subject + "/life.npy")
myfirstdaywiththeyankees = np.load("data/fmri_data/storydata/" + subject + "/myfirstdaywiththeyankees.npy")
naked = np.load("data/fmri_data/storydata/" + subject + "/naked.npy")
odetostepfather = np.load("data/fmri_data/storydata/" + subject + "/odetostepfather.npy")
souls = np.load("data/fmri_data/storydata/" + subject + "/souls.npy")
undertheinfluence = np.load("data/fmri_data/storydata/" + subject + "/undertheinfluence.npy")

# Prep data
ai_fmri, ai_features = prep_data(alternateithicatom, alternateithicatom_transformed)
avatar_fmri, avatar_features = prep_data(avatar, avatar_transformed)
howtodraw_fmri, howtodraw_features = prep_data(howtodraw, howtodraw_transformed)
legacy_fmri, legacy_features = prep_data(legacy, legacy_transformed)
life_fmri, life_features = prep_data(life, life_transformed)
yankees_fmri, yankees_features = prep_data(myfirstdaywiththeyankees, myfirstdaywiththeyankees_transformed)
naked_fmri, naked_features = prep_data(naked, naked_transformed)
odetostepfather_fmri, odetostepfather_features = prep_data(odetostepfather, odetostepfather_transformed)
souls_fmri, souls_features = prep_data(souls, souls_transformed)
undertheinfluence_fmri, undertheinfluence_features = prep_data(undertheinfluence, undertheinfluence_transformed)

# Make predictions
print("Make predictions")
ai_predictions = np.dot(ai_features, vision_encoding)
avatar_predictions = np.dot(avatar_features, vision_encoding)
howtodraw_predictions = np.dot(howtodraw_features, vision_encoding)
legacy_predictions = np.dot(legacy_features, vision_encoding)
life_predictions = np.dot(life_features, vision_encoding)
yankees_predictions = np.dot(yankees_features, vision_encoding)
naked_predictions = np.dot(naked_features, vision_encoding)
odetostepfather_predictions = np.dot(odetostepfather_features, vision_encoding)
souls_predictions = np.dot(souls_features, vision_encoding)
undertheinfluence_predictions = np.dot(undertheinfluence_features, vision_encoding)

print("Calculate correlations")
ai_correlations = calc_correlation(ai_predictions, ai_fmri)
avatar_correlations = calc_correlation(avatar_predictions, avatar_fmri)
howtodraw_correlations = calc_correlation(howtodraw_predictions, howtodraw_fmri)
legacy_correlations = calc_correlation(legacy_predictions, legacy_fmri)
life_correlations = calc_correlation(life_predictions, life_fmri)
yankees_correlations = calc_correlation(yankees_predictions, yankees_fmri)
naked_correlations = calc_correlation(naked_predictions, naked_fmri)
odetostepfather_correlations = calc_correlation(odetostepfather_predictions, odetostepfather_fmri)
souls_correlations = calc_correlation(souls_predictions, souls_fmri)
undertheinfluence_correlations = calc_correlation(undertheinfluence_predictions, undertheinfluence_fmri)

all_correlations = np.stack((ai_correlations, avatar_correlations, howtodraw_correlations,
                             legacy_correlations, life_correlations, yankees_correlations,
                             naked_correlations, odetostepfather_correlations, souls_correlations,
                             undertheinfluence_correlations))

story_correlations = np.nanmean(all_correlations, axis=0)
print("max story correlation:", np.nanmax(story_correlations))

# Recreate the mask used for flattening (assuming you have access to 'movie_train' or similar)
mask = ~np.isnan(alternateithicatom[0])  # Using the first time point as a reference for the mask

# Initialize an empty 3D array with NaNs for the correlation data
story_reconstructed_correlations = np.full((31, 100, 100), np.nan)

# Flatten the mask to get the indices of the original valid (non-NaN) data points
valid_indices = np.where(mask.flatten())[0]

# Assign the correlation coefficients to their original spatial positions
for index, corr_value in zip(valid_indices, story_correlations):
    # Convert the 1D index back to 3D index in the spatial dimensions
    z, x, y = np.unravel_index(index, (31, 100, 100))
    story_reconstructed_correlations[z, x, y] = corr_value

story_flattened_correlations = story_reconstructed_correlations.flatten()

lh_mapping_matrix = load_npz("data/fmri_data/mappers/" + subject + "_listening_forVL_lh.npz")
story_lh_vertex_correlation_data = lh_mapping_matrix.dot(story_flattened_correlations)
lh_vertex_coords = np.load("data/fmri_data/mappers/" + subject + "_vertex_coords_lh.npy")
rh_mapping_matrix = load_npz("data/fmri_data/mappers/" + subject + "_listening_forVL_rh.npz")
story_rh_vertex_correlation_data = rh_mapping_matrix.dot(story_flattened_correlations)
rh_vertex_coords = np.load("data/fmri_data/mappers/" + subject + "_vertex_coords_rh.npy")

vmin, vmax = -0.1, 0.1
fig, axs = plt.subplots(1, 2, figsize=(7,4))

# Plot the first flatmap
sc1 = axs[0].scatter(lh_vertex_coords[:, 0], lh_vertex_coords[:, 1], c=story_lh_vertex_correlation_data, cmap='RdBu_r', vmin=vmin, vmax=vmax, s=.005)
axs[0].set_aspect('equal', adjustable='box')  # Ensure equal scaling
# axs[0].set_title('Left Hemisphere')
axs[0].set_frame_on(False)
axs[0].set_xticks([])  # Remove x-axis ticks
axs[0].set_yticks([])  # Remove y-axis ticks

# Plot the second flatmap
sc2 = axs[1].scatter(rh_vertex_coords[:, 0], rh_vertex_coords[:, 1], c=story_rh_vertex_correlation_data, cmap='RdBu_r', vmin=vmin, vmax=vmax, s=.005)
axs[1].set_aspect('equal', adjustable='box')  # Ensure equal scaling
# axs[1].set_title('Right Hemisphere')
axs[1].set_frame_on(False)
axs[1].set_xticks([])  # Remove x-axis ticks
axs[1].set_yticks([])  # Remove y-axis ticks

# Adjust layout to make space for the top colorbar
plt.subplots_adjust(top=0.85, wspace=0)

# Add a single horizontal colorbar at the top
cbar_ax = fig.add_axes([0.25, 0.9, 0.5, 0.03])  # Adjust these values as needed [left, bottom, width, height]
cbar = fig.colorbar(sc1, cax=cbar_ax, orientation='horizontal')

# Set the color bar to only display min and max values
cbar.set_ticks([vmin, vmax])
cbar.set_ticklabels([f'{vmin}', f'{vmax}'])

# Remove the color bar box
cbar.outline.set_visible(False)

plt.title(r'$r_{\mathit{movie \rightarrow story}}$')
plt.savefig('results/visuals' + subject + '_movie_story.png', format='png')
plt.show()

print("Start story -> movie analysis")
train_nan = np.load("data/fmri_data/moviedata/S1/train.npy")
test_nan = np.load("data/fmri_data/moviedata/S1/test.npy")

train = remove_nan(train_nan)
test = remove_nan(test_nan)

movie_feature_arrays = [train00_transformed, train01_transformed, train02_transformed,
                        train03_transformed, train04_transformed, train05_transformed,
                        train06_transformed, train07_transformed, train08_transformed,
                        train09_transformed, train10_transformed, train11_transformed]

# Combine data
movie_fmri_train = train
movie_fmri_test = test

movie_features_train = np.vstack(movie_feature_arrays)
movie_features_test = test_transformed

print("Make predictions")
movie_predictions_train = np.dot(movie_features_train, language_encoding)
movie_predictions_test = np.dot(movie_features_test, language_encoding)

print("Calculate correlations")
movie_correlations_train = calc_correlation(movie_predictions_train, movie_fmri_train)
movie_correlations_test = calc_correlation(movie_predictions_test, movie_fmri_test)

all_correlations = np.stack((movie_correlations_train, movie_correlations_train, movie_correlations_train,
                             movie_correlations_train, movie_correlations_train, movie_correlations_train,
                             movie_correlations_train, movie_correlations_train, movie_correlations_train,
                             movie_correlations_train, movie_correlations_train, movie_correlations_train,
                             movie_correlations_test))

movie_correlations = story_correlations = np.nanmean(all_correlations, axis=0)
print("Max correlations:", np.nanmax(movie_correlations))

print("Visualize results")
# Recreate the mask used for flattening (assuming you have access to 'movie_train' or similar)
mask = ~np.isnan(train_nan[0])  # Using the first time point as a reference for the mask

# Initialize an empty 3D array with NaNs for the correlation data
movie_reconstructed_correlations = np.full((31, 100, 100), np.nan)

# Flatten the mask to get the indices of the original valid (non-NaN) data points
valid_indices = np.where(mask.flatten())[0]

# Assign the correlation coefficients to their original spatial positions
for index, corr_value in zip(valid_indices, movie_correlations):
    # Convert the 1D index back to 3D index in the spatial dimensions
    z, x, y = np.unravel_index(index, (31, 100, 100))
    movie_reconstructed_correlations[z, x, y] = corr_value

movie_flattened_correlations = movie_reconstructed_correlations.flatten()
movie_lh_vertex_correlation_data = lh_mapping_matrix.dot(movie_flattened_correlations)
movie_rh_vertex_correlation_data = rh_mapping_matrix.dot(movie_flattened_correlations)

vmin, vmax = -0.1, 0.1
fig, axs = plt.subplots(1, 2, figsize=(7,4))

# Plot the first flatmap
sc1 = axs[0].scatter(lh_vertex_coords[:, 0], lh_vertex_coords[:, 1], c=movie_lh_vertex_correlation_data, cmap='RdBu_r', vmin=vmin, vmax=vmax, s=.02)
axs[0].set_aspect('equal', adjustable='box')  # Ensure equal scaling
# axs[0].set_title('Left Hemisphere')
axs[0].set_frame_on(False)
axs[0].set_xticks([])  # Remove x-axis ticks
axs[0].set_yticks([])  # Remove y-axis ticks

# Plot the second flatmap
sc2 = axs[1].scatter(rh_vertex_coords[:, 0], rh_vertex_coords[:, 1], c=movie_rh_vertex_correlation_data, cmap='RdBu_r', vmin=vmin, vmax=vmax, s=.02)
axs[1].set_aspect('equal', adjustable='box')  # Ensure equal scaling
# axs[1].set_title('Right Hemisphere')
axs[1].set_frame_on(False)
axs[1].set_xticks([])  # Remove x-axis ticks
axs[1].set_yticks([])  # Remove y-axis ticks

# Adjust layout to make space for the top colorbar
plt.subplots_adjust(top=0.85, wspace=0)

# Add a single horizontal colorbar at the top
cbar_ax = fig.add_axes([0.25, 0.9, 0.5, 0.05])  # Adjust these values as needed [left, bottom, width, height]
cbar = fig.colorbar(sc1, cax=cbar_ax, orientation='horizontal')
# Set the color bar to only display min and max values
cbar.set_ticks([vmin, vmax])
cbar.set_ticklabels([f'{vmin}', f'{vmax}'])

# Remove the color bar box
cbar.outline.set_visible(False)

plt.title(r'$r_{\mathit{story \rightarrow movie}}$')
plt.savefig('results/visuals/' + subject + 'story_movie.png', format='png')
plt.show()