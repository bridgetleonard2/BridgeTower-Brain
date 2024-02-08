import numpy as np
# We will be using L2-regularized linear regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, RandomizedSearchCV
from joblib import Parallel, delayed
from tqdm import tqdm  # Progress bar

import warnings

print("Load movie data")
# Load movie data
s1_movie_train = np.load("data/moviedata/S1/train.npy")

mask = ~np.isnan(s1_movie_train)

# Apply the mask and then flatten
# This will keep only the non-NaN values
s1_movie_fmri = s1_movie_train[mask].reshape(s1_movie_train.shape[0], -1)

print("Load movie features")
# Load in movie feature vectors
# movie data
train00 = np.load("data/feature_vectors/movie/train_00_data.npy")
train01 = np.load("data/feature_vectors/movie/train_01_data.npy")
train02 = np.load("data/feature_vectors/movie/train_02_data.npy")
train03 = np.load("data/feature_vectors/movie/train_03_data.npy")
train04 = np.load("data/feature_vectors/movie/train_04_data.npy")
train05 = np.load("data/feature_vectors/movie/train_05_data.npy")
train06 = np.load("data/feature_vectors/movie/train_06_data.npy")
train07 = np.load("data/feature_vectors/movie/train_07_data.npy")
train08 = np.load("data/feature_vectors/movie/train_08_data.npy")
train09 = np.load("data/feature_vectors/movie/train_09_data.npy")
train10 = np.load("data/feature_vectors/movie/train_10_data.npy")
train11 = np.load("data/feature_vectors/movie/train_11_data.npy")

movie_features = np.vstack((train00, train01, train02, train03, train04,
                            train05, train06, train07, train08, train09,
                            train10, train11))

# Finite impulse response delays
delays = [2, 4, 6, 8]  # Delays in seconds
shifted_features_list = []

for delay in delays:
    shift_amount = delay // 2  # Assuming TR is 2 seconds
    shifted = np.roll(movie_features, shift_amount, axis=0)
    # Optionally, handle edge effects here (e.g., zero-padding or trimming)
    shifted_features_list.append(shifted)

# Stack the shifted arrays to create a 3D array
delayed_features = np.stack(shifted_features_list, axis=-1)

# Reshape the feature data for regression
n_time_points, n_features, n_delays = delayed_features.shape
features_reshaped = delayed_features.reshape(n_time_points,
                                             n_features * n_delays)

n_voxels = s1_movie_fmri.shape[1]
alphas = np.logspace(-6, 6, 50)  # Range for alphas
# Use a smaller number of folds
kf = KFold(n_splits=3, shuffle=True, random_state=42)

best_alphas = np.zeros(1600)
coefficients = np.zeros((1600, features_reshaped.shape[1]))


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

# Split voxels into chunks
num_cpus = 8  # Change to 16 if using 16 CPUs
voxel_start = 4867
num_voxels = 1600
voxel_end = voxel_start + num_voxels + 1  # Add 1 to include voxel 6567

voxels_per_chunk = num_voxels // num_cpus
voxel_chunks = []

for i in range(voxel_start, voxel_end, voxels_per_chunk):
    chunk_end = min(i + voxels_per_chunk, voxel_end)
    voxel_chunks.append(list(range(i, chunk_end)))

# Process each chunk in parallel
results = Parallel(n_jobs=num_cpus, backend="loky")(
    delayed(process_voxel_chunk)(chunk, features_reshaped, s1_movie_fmri, alphas, kf)
    for chunk in voxel_chunks
)

# Flatten the results list
results = [result for chunk_results in results for result in chunk_results]

# Update best_alphas and coefficients
for voxel, alpha, coef in results:
    best_alphas[voxel] = alpha
    coefficients[voxel] = coef

# Save results
np.save('results/movie/best_alphas_4867-6467.npy', best_alphas)
np.save('results/movie/coefficients_4867-6467.npy', coefficients)