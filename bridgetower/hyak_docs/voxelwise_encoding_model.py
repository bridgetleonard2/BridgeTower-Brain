import numpy as np
# We will be using L2-regularized linear regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import warnings


# Load movie data
s1_movie_test = np.load("/gscratch/scrubbed/bll313/data/moviedata/S1 \
                        /test.npy")
s1_movie_train = np.load("/gscratch/scrubbed/bll313/data/moviedata/S1 \
                         /train.npy")

# Assuming your data is in a NumPy array with shape (n_tr, z, x, y)
# Creating a mask where the data is not NaN
mask = ~np.isnan(s1_movie_test)

# Apply the mask and then flatten
# This will keep only the non-NaN values
s1_movie_test_flat = s1_movie_test[mask].reshape(s1_movie_test.shape[0], -1)

mask = ~np.isnan(s1_movie_train)

# Apply the mask and then flatten
# This will keep only the non-NaN values
s1_movie_train_flat = s1_movie_train[mask].reshape(s1_movie_train.shape[0], -1)

# Combine final flattened movie data
s1_movie_fmri = np.concatenate((s1_movie_train_flat, s1_movie_test_flat),
                               axis=0)

# Load in movie feature vectors
# movie data
test = np.load("/gscratch/scrubbed/bll313/data/feature_vectors/movie \
               /test_data.npy")
train00 = np.load("/gscratch/scrubbed/bll313/data/feature_vectors/movie \
                  /train_00_data.npy")
train01 = np.load("/gscratch/scrubbed/bll313/data/feature_vectors/movie \
                  /train_01_data.npy")
train02 = np.load("/gscratch/scrubbed/bll313/data/feature_vectors/movie \
                  /train_02_data.npy")
train03 = np.load("/gscratch/scrubbed/bll313/data/feature_vectors/movie \
                  /train_03_data.npy")
train04 = np.load("/gscratch/scrubbed/bll313/data/feature_vectors/movie \
                  /train_04_data.npy")
train05 = np.load("/gscratch/scrubbed/bll313/data/feature_vectors/movie \
                  /train_05_data.npy")
train06 = np.load("/gscratch/scrubbed/bll313/data/feature_vectors/movie \
                  /train_06_data.npy")
train07 = np.load("/gscratch/scrubbed/bll313/data/feature_vectors/movie \
                  /train_07_data.npy")
train08 = np.load("/gscratch/scrubbed/bll313/data/feature_vectors/movie \
                  /train_08_data.npy")
train09 = np.load("/gscratch/scrubbed/bll313/data/feature_vectors/movie \
                  /train_09_data.npy")
train10 = np.load("/gscratch/scrubbed/bll313/data/feature_vectors/movie \
                  /train_10_data.npy")
train11 = np.load("/gscratch/scrubbed/bll313/data/feature_vectors/movie \
                  /train_11_data.npy")

movie_features = np.vstack((train00, train01, train02, train03, train04,
                            train05, train06, train07, train08, train09,
                            train10, train11, test))

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

warnings.filterwarnings("ignore")


def test():
    for voxel in range(5):
        ridge_cv = RidgeCV(alphas=alphas, cv=kf)
        ridge_cv.fit(features_reshaped, s1_movie_fmri[:, voxel])
        print("Voxel", voxel, "best alpha:", ridge_cv.alpha_)
        print("Voxel", voxel, "first ten coefficients:", ridge_cv.coef_)


test()

for voxel in range(n_voxels):
    ridge_cv = RidgeCV(alphas=alphas, cv=kf)
    ridge_cv.fit(features_reshaped, s1_movie_fmri[:, voxel])
    best_alphas[voxel] = ridge_cv.alpha_
    coefficients[voxel, :] = ridge_cv.coef_

    # alphas = np.array(best_alphas)
    # coef = np.array(coefficients)
    np.save('/gscratch/scrubbed/bll313/results/movie/best_alphas.npy',
            best_alphas)
    np.save('/gscratch/scrubbed/bll313/results/movie/coefficients.npy',
            coefficients)
    # Print checkpoints
    if voxel == round(n_voxels*0.1):
        print("10% done")
    elif voxel == round(n_voxels*0.2):
        print("20% done")
    elif voxel == round(n_voxels*0.3):
        print("30% done")
    elif voxel == round(n_voxels*0.4):
        print("40% done")
    elif voxel == round(n_voxels*0.5):
        print("50% done")
    elif voxel == round(n_voxels*0.6):
        print("60% done")
    elif voxel == round(n_voxels*0.7):
        print("70% done")
    elif voxel == round(n_voxels*0.8):
        print("80% done")
    elif voxel == round(n_voxels*0.9):
        print("90% done")
print("Complete")
