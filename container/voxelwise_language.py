import numpy as np
# We will be using L2-regularized linear regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, RandomizedSearchCV
from joblib import Parallel, delayed
from scipy.signal import resample
from tqdm import tqdm  # Progress bar

import warnings

print("Load story data")
# Load movie data
s1_alternateithicatom = np.load("data/storydata/S1/alternateithicatom.npy")
s1_avatar = np.load("data/storydata/S1/avatar.npy")
s1_howtodraw = np.load("data/storydata/S1/howtodraw.npy")
s1_legacy = np.load("data/storydata/S1/legacy.npy")
s1_life = np.load("data/storydata/S1/life.npy")
s1_myfirstdaywiththeyankees = np.load("data/storydata/S1/myfirstdaywiththeyankees.npy")
s1_naked = np.load("data/storydata/S1/naked.npy")
s1_odetostepfather = np.load("data/storydata/S1/odetostepfather.npy")
s1_souls = np.load("data/storydata/S1/souls.npy")

print("Load story features")
alternateithicatom = np.load("data/feature_vectors/story/alternateithicatom_data.npy")
avatar = np.load("data/feature_vectors/story/avatar_data.npy")
howtodraw = np.load("data/feature_vectors/story/howtodraw_data.npy")
legacy = np.load("data/feature_vectors/story/legacy_data.npy")
life = np.load("data/feature_vectors/story/life_data.npy")
myfirstdaywiththeyankees = np.load("data/feature_vectors/story/myfirstdaywiththeyankees_data.npy")
naked = np.load("data/feature_vectors/story/naked_data.npy")
odetostepfather = np.load("data/feature_vectors/story/odetostepfather_data.npy")
souls = np.load("data/feature_vectors/story/souls_data.npy")

def remove_nan(data):
    mask = ~np.isnan(data)

    # Apply the mask and then flatten
    # This will keep only the non-NaN values
    data_reshaped = data[mask].reshape(data.shape[0], -1)

    print("fMRI shape:", data_reshaped.shape)
    return data_reshaped


def resample_to_acq(feature_data, fmri_data):
    dimensions = fmri_data.shape[0]
    data_transposed = feature_data.T
    data_resampled = np.empty((data_transposed.shape[0], dimensions))

    for i in range(data_transposed.shape[0]):
        data_resampled[i, :] = resample(data_transposed[i, :], dimensions, window=('kaiser', 14))

    print("Shape after resampling:", data_resampled.T.shape)
    return data_resampled.T


def delay_features(features):
    delays = [2, 4, 6, 8]  # Delays in seconds
    shifted_features_list = []

    for delay in delays:
        shift_amount = delay // 2  # Assuming TR is 2 seconds
        shifted = np.roll(features, shift_amount, axis=0)
        # Optionally, handle edge effects here (e.g., zero-padding or trimming)
        shifted_features_list.append(shifted)

    # Stack the shifted arrays to create a 3D array
    shifted_features_3d = np.stack(shifted_features_list, axis=-1)

    # Reshape the feature data for regression
    n_time_points, n_features, n_delays = shifted_features_3d.shape
    features_reshaped = shifted_features_3d.reshape(n_time_points, n_features * n_delays)

    print("Shape after delays:", features_reshaped.shape)
    return features_reshaped


def prep_data(fmri_data, feature_data):
    fmri_reshaped = remove_nan(fmri_data)

    feature_resampled = resample_to_acq(feature_data, fmri_reshaped)
    feature_delayed = delay_features(feature_resampled)

    return fmri_reshaped, feature_delayed


s1_ai_fmri, s1_ai_features = prep_data(s1_alternateithicatom, alternateithicatom)
s1_avatar_fmri, s1_avatar_features = prep_data(s1_avatar, avatar)
s1_howtodraw_fmri, s1_howtodraw_features = prep_data(s1_howtodraw, howtodraw)
s1_life_fmri, s1_life_features = prep_data(s1_life, life)
s1_yankees_fmri, s1_yankees_features = prep_data(s1_myfirstdaywiththeyankees, myfirstdaywiththeyankees)
s1_naked_fmri, s1_naked_features = prep_data(s1_naked, naked)
s1_ode_fmri, s1_ode_features = prep_data(s1_odetostepfather, odetostepfather)
s1_souls_fmri, s1_souls_features = prep_data(s1_souls, souls)


# Function to process a chunk of voxels
def process_voxel_chunk(voxel_chunk, features_reshaped, fmri_data, alphas, kf):
    results = []
    for voxel in tqdm(voxel_chunk, desc="Processing chunk"):
        result = process_voxel(voxel, features_reshaped, fmri_data, alphas, kf)
        results.append(result)
    return results


# Function to process a single voxel
def process_voxel(voxel, features_reshaped, fmri_data, alphas, kf):
    warnings.filterwarnings("ignore")
    ridge_cv = RidgeCV(alphas=alphas, cv=kf)
    ridge_cv.fit(features_reshaped, fmri_data[:, voxel])
    return voxel, ridge_cv.alpha_, ridge_cv.coef_


n_voxels = s1_ai_fmri.shape[1]
alphas = np.logspace(-6, 6, 50)  # Range for alphas
# Use a smaller number of folds
kf = KFold(n_splits=3, shuffle=True, random_state=42)

best_alphas = np.zeros(1600)
coefficients = np.zeros((1600, 3072))

### CHOOSE PARAM HERE
start_batch = 0
features = s1_ai_features
fmri = s1_ai_fmri
title = "ai_"

for i in range(40):
    # Split voxels into chunks
    num_cpus = 16  # Change to 16 if using 16 CPUs
    num_voxels = 1600
    voxel_start = start_batch + (i * num_voxels)
    voxel_end = voxel_start + num_voxels

    voxels_per_chunk = num_voxels // num_cpus
    voxel_chunks = []

    for i in range(voxel_start, voxel_end, voxels_per_chunk):
        chunk_end = min(i + voxels_per_chunk, voxel_end)
        voxel_chunks.append(list(range(i, chunk_end)))


    # Process each chunk in parallel
    results = Parallel(n_jobs=num_cpus, backend="loky")(
        delayed(process_voxel_chunk)(chunk, features, fmri, alphas, kf)
        for chunk in voxel_chunks
    )

    # Flatten the results list
    results = [result for chunk_results in results for result in chunk_results]

    # Update best_alphas and coefficients
    for voxel, alpha, coef in results:
        best_alphas[voxel - voxel_start] = alpha
        coefficients[voxel - voxel_start] = coef

    filename_alphas = 'results/story/best_alphas_' + title + str(voxel_start) + '_' + str(voxel_end - 1) + '.npy'
    filename_coefficients = 'results/story/coefficients_' + title + str(voxel_start) + '_' + str(voxel_end - 1) + '.npy'
    
    # Save results
    np.save(filename_alphas, best_alphas)
    np.save(filename_coefficients, coefficients)
