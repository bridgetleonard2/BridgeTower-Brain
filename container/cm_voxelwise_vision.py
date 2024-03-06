import numpy as np
from sklearn.model_selection import check_cv
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import set_config
from functions import remove_nan
from functions import generate_leave_one_run_out
from functions import Delayer
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.backend import set_backend


# CHOOSE SUBJECT
subject = 'S1'

# Load data
print("Load movie data")
# Load fMRI data
# Using all data for cross-modality encoding model
fmri_train = np.load("data/moviedata/S1/train.npy")
fmri_test = np.load("data/moviedata/S1/train.npy")

print("Load movie features")
# Load in movie feature vectors
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
test = np.load("data/feature_vectors/movie/test_data.npy")

# Prep data
train_fmri = remove_nan(fmri_train)
test_fmri = remove_nan(fmri_test)

fmri_arrays = [train_fmri, test_fmri]
feature_arrays = [train00, train01, train02, train03, train04,
                  train05, train06, train07, train08, train09,
                  train10, train11, test]

# Combine data
Y_train = np.vstack(fmri_arrays)
X_train = np.vstack(feature_arrays)

