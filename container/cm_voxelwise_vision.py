import numpy as np
import functions

# Load data
print("Load movie data")
# Load fMRI data
s1_movie_train = np.load("data/moviedata/S1/train.npy")
s1_movie_fmri = remove_nan(s1_movie_train)

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

movie_features = np.vstack((train00, train01, train02, train03, train04,
                            train05, train06, train07, train08, train09,
                            train10, train11))