import numpy as np
# We will be using L2-regularized linear regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
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
kf = KFold(n_splits=5)  # Example of 5-fold cross-validation

best_alphas = np.zeros(n_voxels)
coefficients = np.zeros((n_voxels, features_reshaped.shape[1]))


def process_voxel(voxel, features_reshaped, fmri_data, alphas, kf):
    warnings.filterwarnings("ignore")

    ridge_cv = RidgeCV(alphas=alphas, cv=kf)
    ridge_cv.fit(features_reshaped, fmri_data[:, voxel])
    return ridge_cv.alpha_, ridge_cv.coef_


print("Test with voxel 1")
test_alpha, test_coef = process_voxel(0, features_reshaped, s1_movie_fmri, alphas, kf)
print("Voxel 1:", test_alpha, test_coef[:5])

print("Running all voxels...")
# Parallel processing
results = Parallel(n_jobs=8, backend="loky")(delayed(process_voxel)(voxel, features_reshaped, s1_movie_fmri, alphas, kf) for voxel in tqdm(range(n_voxels)))

# Extract results
best_alphas, coefficients = zip(*results)

# Save results
np.save('results/movie/best_alphas.npy', best_alphas)
np.save('results/movie/coefficients.npy', coefficients)

print("Complete")
